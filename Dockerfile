ARG from=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
# Use 'runtime' only if you want no 'flash-attention'
# ARG from=nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# FROM 
FROM ${from} as base

###### options to build the image
# BUNDLE_REQUIREMENTS: whether to install requirements.txt when docker build
# BUNDLE_FLASH_ATTENTION: whether to install flash-attention when docker build

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

RUN <<EOF
if [ "$BUNDLE_REQUIREMENTS" = "true" ]; then 
    cd /data/shared/Qwen-7B
    pip3 install -r requirements.txt
    pip3 install gradio mdtex2html scipy argparse
fi
EOF

FROM bundle_req as bundle_flash_attention
ARG BUNDLE_FLASH_ATTENTION=false

RUN <<EOF 
if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then
    cd /data/shared/Qwen-7B 
    test -d flash-attention || git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
    cd /data/shared/Qwen-7B/flash-attention &&
        pip3 install . &&
        pip3 install csrc/layer_norm &&
        pip3 install csrc/rotary
    fi
EOF

FROM bundle_flash_attention as final

ARG from

EXPOSE 80

WORKDIR /data/shared/Qwen-7B/

COPY <<EOF ./run_web_demo.sh
#!/usr/bin/env bash

thisDir=\$(pwd)

FROM_IMAGE=$from # from base image
export INSTALL_DEPS=true
export INSTALL_FLASH_ATTN=\$(test "\${FROM_IMAGE#*devel}" != "\$FROM_IMAGE" && echo true)
declare -a WEB_DEMO_ARGS=(--server-port 80 --server-name 0.0.0.0 --inbrowser)


echo "INSTALL_DEPS: \$INSTALL_DEPS"
echo "INSTALL_FLASH_ATTN: \$INSTALL_FLASH_ATTN"
echo "WEB_DEMO_ARGS: \${WEB_DEMO_ARGS[@]}"

function performInstall() {

    pushd "\$thisDir"
    pip3 install -r requirements.txt
    pip3 install gradio mdtex2html scipy argparse

    if \$INSTALL_FLASH_ATTN; then
        if [[ ! -d flash-attention ]]; then
            if ! git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention; then
                echo "Clone flash-attention failed, please install it manually."
                return 0
            fi
        fi

        cd flash-attention &&
            pip3 install . &&
            pip3 install csrc/layer_norm &&
            pip3 install csrc/rotary ||
            echo "Install flash-attention failed, please install it manually."
    fi

    popd
}

echo "Starting WebUI..."

if ! python3 web_demo.py \${WEB_DEMO_ARGS[@]}; then
    if \$INSTALL_DEPS; then
        echo "Installing deps, and try again..."
        performInstall && python3 web_demo.py \${WEB_DEMO_ARGS[@]}
    else
        echo "Please install deps manually, or use --install-deps to install deps automatically."
    fi
fi
EOF

# See the result when building the image
RUN cat run_web_demo.sh

CMD ["bash", "run_web_demo.sh"]

############  Usage ############

## Build the image (sample)
# docker build -t qwen-7b .

## Run the image (sample)
#  docker run --gpus all -d --restart always --name qwen-7b \
#        -v /var/run/docker.sock:/var/run/docker.sock \
#        -p 60080:80 \
#        -it \
#        qwen-7b:latest