#!/usr/bin/env bash

cd "$(dirname "$0")"
thisDir=$(pwd)

function performInstall() {
    set -e

    pushd "$thisDir"
    pip3 install -r requirements.txt
    pip3 install gradio mdtex2html scipy

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
    popd
}

echo "Starting WebUI..."

if ! python3 web_demo.py; then
    echo "Run demo failed, install the deps and try again? (y/n)"
    # auto perform install if in docker
    if [[ -t 0 ]] && [[ -t 1 ]] && [[ ! -f "/.dockerenv" ]]; then
        read doInstall
    else
        doInstall="y"
    fi

    if ! [[ "$doInstall" =~ y|Y ]]; then
        exit 1
    fi

    echo "Installing deps, and try again..."
    performInstall && python3 web_demo.py
fi
