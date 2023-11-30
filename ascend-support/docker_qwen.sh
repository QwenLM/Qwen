#!/bin/bash

IMAGE_NAME=qwenllm/qwen-mindspore:v23.0.RC3
CONTAINER_NAME=qwen-mindspore
CHECKPOINT_PATH='NOT_SET'

DOCKER_CHECKPOINT_PATH=/data/qwen/models/Qwen-7B-Chat

function usage() {
    echo '
Usage: bash ascend-support/docker_qwen.sh [-i IMAGE_NAME] -c [/path/to/Qwen-7B-Chat] [-n CONTAINER_NAME]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -i | --image )
            shift
            IMAGE_NAME=$1
            ;;
        -c | --checkpoint )
            shift
            CHECKPOINT_PATH=$1
            ;;
        -n | --name )
            shift
            CONTAINER_NAME=$1
            ;;
        -h )
            usage
            exit
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

docker run -it --rm -u root --network=host --ipc=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --name=${CONTAINER_NAME} \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v ${CHECKPOINT_PATH}:${DOCKER_CHECKPOINT_PATH} \
    -v /var/log/npu/:/usr/slog \
    ${IMAGE_NAME} /bin/bash
