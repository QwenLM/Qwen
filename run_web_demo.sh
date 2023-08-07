#!/usr/bin/env bash

cd "$(dirname "$0")"
thisDir=$(pwd)

export INSTALL_DEPS=false
export INSTALL_FLASH_ATTN=false

declare -a PASS_THROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
    -h | --help)
        echo "Usage: $0 [-h|--help] [--install-deps] [--install-flash-attn]"
        exit 0
        ;;
    --install-deps)
        export INSTALL_DEPS=true
        shift
        ;;
    --install-flash-attn)
        export INSTALL_FLASH_ATTN=true
        shift
        ;;
    -)
        shift
        PASS_THROUGH_ARGS=($@)
        break
        ;;

    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

echo "INSTALL_DEPS: $INSTALL_DEPS"
echo "INSTALL_FLASH_ATTN: $INSTALL_FLASH_ATTN"
echo "PASS_THROUGH_ARGS: ${PASS_THROUGH_ARGS[@]}"

function performInstall() {

    pushd "$thisDir"
    pip3 install -r requirements.txt
    pip3 install gradio mdtex2html scipy argparse

    if $INSTALL_FLASH_ATTN; then
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

if ! python3 web_demo.py ${PASS_THROUGH_ARGS[@]}; then
    if $INSTALL_DEPS; then
        echo "Installing deps, and try again..."
        performInstall && python3 web_demo.py ${PASS_THROUGH_ARGS[@]}
    else
        echo "Please install deps manually, or use --install-deps to install deps automatically."
    fi
fi
