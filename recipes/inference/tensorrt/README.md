# Inference Qwen Using TensorRT-LLM
Below, we provide a simple example to show how to inference Qwen by TensorRT-LLM. We recommend using GPUs with compute capability of at least SM_80 such as A10 and A800 to run this example, as we have tested on these GPUs. You can find your gpu compute capability on this [link](https://developer.nvidia.com/cuda-gpus).

## Installation
You can use pre-built docker image to run this example. Simultaneously, You can also refer to the official [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for installation and detailed usage.
```bash
docker run --gpus all -it --ipc=host --network=host pai-image-manage-registry.cn-wulanchabu.cr.aliyuncs.com/pai/llm-inference:tensorrt-llm-0.8.0 bash
```
## Quickstart
1. Download model by modelscope

```bash
cd TensorRT-LLM/examples/qwen
python3 -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('Qwen/Qwen-1_8B-Chat', cache_dir='.', revision='master')"
mkdir -p ./tmp/Qwen
mv Qwen/Qwen-1_8B-Chat ./tmp/Qwen/1_8B
```

2. Build TensorRT engine from HF checkpoint

```bash
python3 build.py --hf_model_dir ./tmp/Qwen/1_8B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/1_8B/trt_engines/fp16/1-gpu/
```

3. Inference
```bash
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=512 \
                  --tokenizer_dir ./tmp/Qwen/1_8B/ \
                  --engine_dir=./tmp/Qwen/1_8B/trt_engines/fp16/1-gpu
```
```
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "你好，我是来自阿里云的大规模语言模型，我叫通义千问。"
```
