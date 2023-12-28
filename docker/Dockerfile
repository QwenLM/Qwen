ARG CUDA_VERSION=11.7.1
ARG from=nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu20.04

FROM ${from} as base

ARG from

RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
&& rm -rf /var/lib/apt/lists/*
EOF

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git lfs install

FROM base as dev

WORKDIR /

RUN mkdir -p /data/shared/Qwen

WORKDIR /data/shared/Qwen/

# Users can also mount '/data/shared/Qwen/' to keep the data
COPY ../requirements.txt ./
COPY ../requirements_web_demo.txt ./

FROM dev as bundle_req

ARG BUNDLE_REQUIREMENTS=true

RUN <<EOF
if [ "$BUNDLE_REQUIREMENTS" = "true" ]; then 
    cd /data/shared/Qwen
    pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
    pip3 install -r requirements.txt
    pip3 install -r requirements_web_demo.txt
fi
EOF

FROM bundle_req as bundle_flash_attention
ARG BUNDLE_FLASH_ATTENTION=true

RUN <<EOF 
if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then
    cd /data/shared/Qwen 
    test -d flash-attention || git clone -b v2.3.3 https://github.com/Dao-AILab/flash-attention
    cd /data/shared/Qwen/flash-attention &&
        pip3 install . &&
        pip3 install csrc/layer_norm
fi
EOF

FROM bundle_flash_attention as bundle_finetune
ARG BUNDLE_FINETUNE=true

RUN <<EOF
if [ "$BUNDLE_FINETUNE" = "true" ]; then
    cd /data/shared/Qwen

    # Full-finetune / LoRA.
    pip3 install deepspeed "peft==0.5.0"

    # Q-LoRA.
    apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        libopenmpi-dev openmpi-bin \
        && rm -rf /var/lib/apt/lists/*
    pip3 install "optimum==1.12.0" "auto-gptq==0.4.2" mpi4py
fi
EOF

FROM bundle_finetune as bundle_openai_api
ARG BUNDLE_OPENAI_API=true

RUN <<EOF
if [ "$BUNDLE_OPENAI_API" = "true" ]; then
    cd /data/shared/Qwen

    pip3 install fastapi uvicorn "openai<1.0.0" sse_starlette "pydantic<=1.10.13"
fi
EOF

FROM bundle_openai_api as final
ARG from

COPY ../requirements.txt ./
COPY ../requirements_web_demo.txt ./
COPY ../cli_demo.py ./
COPY ../web_demo.py ./
COPY ../openai_api.py ./
COPY ../finetune.py ./
COPY ../utils.py ./
COPY ./examples/* ./examples/
COPY ./eval/* ./eval/
COPY ./finetune/* ./finetune/

EXPOSE 80

WORKDIR /data/shared/Qwen/

CMD ["python3", "web_demo.py", "--server-port", "80", "--server-name", "0.0.0.0", "-c", "/data/shared/Qwen/Qwen-Chat/"]
