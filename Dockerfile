ARG from=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
# Use 'runtime' only if you want no 'flash-attention'
# ARG from=nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# FROM 
FROM ${from} as base

###### options to build the image

ARG from

RUN apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    $(test "${from#*devel}" != "$from" && echo "python3-dev") \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

FROM base as dev

WORKDIR /

RUN mkdir -p /data/shared/Qwen-7B

WORKDIR /data/shared/Qwen-7B/

# Users can also mount '/data/shared/Qwen-7B/' to keep the data
ADD . ./

FROM dev as bundle_req

ARG BUNDLE_REQUIREMENTS=false

RUN if [ "$BUNDLE_REQUIREMENTS" = "true" ]; then \
    cd /data/shared/Qwen-7B; \
    pip3 install -r requirements.txt; \
    pip3 install gradio mdtex2html scipy argparse; \
    fi

FROM bundle_req as bundle_flash_attention
ARG BUNDLE_FLASH_ATTENTION=false

RUN if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
    cd /data/shared/Qwen-7B; \
    test -d flash-attention || git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention; \
    cd /data/shared/Qwen-7B/flash-attention; \
    pip3 install .; \
    pip3 install csrc/layer_norm; \
    pip3 install csrc/rotary; \
    fi

FROM bundle_flash_attention as bundle_models

# the revision of the models to be bundled
ARG BUNDLE_MODELS_REVISION="None"

RUN if [ -n "$BUNDLE_MODELS_REVISION" ] && [ "$BUNDLE_MODELS_REVISION" != "None" ]; then \
    cd /data/shared/Qwen-7B; \
    python3 web_demo.py --exit --model_revision $BUNDLE_MODELS_REVISION; \
    fi

FROM bundle_models as final

ARG BUNDLE_MODELS_REVISION="None"
ARG from

EXPOSE 80

WORKDIR /data/shared/Qwen-7B/
RUN echo "bash ./run_web_demo.sh --install-deps $(test "${from#*devel}" != "$from" && echo --install-flash-attn) - --server_port 80 --server_name 0.0.0.0 --inbrowser --model_revision $BUNDLE_MODELS_REVISION" > /data/shared/Qwen-7B/run.sh

# See the result when building the image
RUN cat /data/shared/Qwen-7B/run.sh

CMD ["bash", "/data/shared/Qwen-7B/run.sh"]

############  Usage ############

## Build the image (sample)
# docker build -t qwen-7b .

## Run the image (sample)
#  docker run --gpus all -d --restart always --name qwen-7b \
#        -v /var/run/docker.sock:/var/run/docker.sock \
#        -p 60080:80 \
#        -it \
#        qwen-7b:latest