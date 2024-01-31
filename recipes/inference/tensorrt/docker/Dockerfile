FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install tensorrt_llm==0.8.0.dev2024011601 -U --no-cache-dir --pre --extra-index-url https://pypi.nvidia.com

RUN pip install --no-cache-dir modelscope==1.11.1

RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
    cd TensorRT-LLM && \ 
    git checkout c89653021e66ca78c55f02b366f404455bc12e8d && \
    pip install --no-cache-dir -r examples/qwen/requirements.txt