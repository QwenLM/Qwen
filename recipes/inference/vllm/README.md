# Inference Qwen Using vLLM

For deployment and fast inference, we suggest using vLLM. 

## Installation

If you use cuda 12.1 and pytorch 2.1, you can directly use the following command to install vLLM.
```bash
# Install vLLM with CUDA 12.1.
pip install vllm
```
Otherwise, please refer to the official vLLM [Installation Instructions](https://docs.vllm.ai/en/latest/getting_started/installation.html).

If you have trouble building vLLM, we recommend using Docker image.

```bash
docker run --gpus all -it --rm --ipc=host --network=host qwenllm/qwen:cu121 bash
```

## GPU Requirements

Qwen model use Bfloat16 by default, but Bfloat16 is only supported on GPUs with compute capability of at least 8. For GPUs with compute capability less than 8.0, it is recommended to set the dtype to float16. You can find your gpu compute capability on this [link](https://developer.nvidia.com/cuda-gpus).

We have tested the GPU memory usage on NVIDIA Tesla V100 32GB by manually adjusting gpu-memory-utilization in eager mode, you can refer to the following table to determine whether your machine is capable of running these models.
| Model | seq_len 2048 | seq_len 8192 | seq_len 16384 | seq_len 32768 |
| :--- | ---: | ---: | ---: | ---: |
| Qwen-1.8B | 6.22G | 7.46G |  |  |
| Qwen-7B | 17.94G | 20.96G |  |  |
| Qwen-7B-Int4 | 9.10G | 12.26G |  |  |
| Qwen-14B | 33.40G |  |  |  |
| Qwen-14B-Int4 | 13.30G |  |  |  |
| Qwen-72B | 166.87G | 185.50G | 210.80G | 253.80G |
| Qwen-72B-int4 | 55.37G | 73.66G | 97.79G | 158.80G |

We have also listed the models that can run on consumer graphics cards by default sequence length in the following table. If the GPU memory only exceeds the model's memory usage by a small margin, you can make the model run on your machine by reducing the max-model-len parameter.</br>
(ps: To run Qwen-14B-Int4 on NVIDIA RTX 3080Ti, you need to set gpu-memory-utilization as 0.99 and enforce eager mode)

| GPU Memory | GPU | Support Model |
| :---: | :---: | :---: |
| 24GB | NVIDIA RTX 4090/3090/A5000 | Qwen-1.8B/Qwen-7B/Qwen-7B-Int4/Qwen-14B-Int4  |
| 16GB | NVIDIA RTX A4000 | Qwen-1.8B/Qwen-7B-Int4/Qwen-14B-Int4 |
| 12GB | NVIDIA RTX 3080Ti/TITAN Xp | Qwen-1.8B/Qwen-14B-Int4 |
| 11GB | NVIDIA RTX 2080Ti/GTX 1080Ti | Qwen-1.8B |
| 10GB | NVIDIA RTX 3080 | Qwen-1.8B |

## Usage

### vLLM + Web Demo / OpenAI-like API

You can use FastChat to launch a web demo or an OpenAI API server. First, install FastChat:

```bash
pip install "fschat[model_worker,webui]=0.2.33" "openai<1.0"
```

To run Qwen with vLLM and FastChat, you need launch a controller by:
```bash
python -m fastchat.serve.controller
```

Then you can launch the model worker, which means loading your model for inference. For single GPU inference, you can directly run:
```bash
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --dtype bfloat16
# run int4 model or GPUs with compute capability less than 8.0
# python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --dtype float16 
```

However, if you hope to run the model on multiple GPUs for faster inference or larger memory, you can use tensor parallelism supported by vLLM. Suppose you run the model on 4 GPUs, the command is shown below:
```bash
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --tensor-parallel-size 4 --dtype bfloat16
# run int4 model or GPUs with compute capability less than 8.0
# python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --tensor-parallel-size 4 --dtype float16 
```

After launching your model worker, you can launch a:

* Web UI Demo
```bash
python -m fastchat.serve.gradio_web_server
```

* OpenAI API
```bash
python -m fastchat.serve.openai_api_server --host localhost --port 8000
```

For OpenAI API server, you can invoke the server in the following manner.

```python
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

# create a request activating streaming response
for chunk in openai.ChatCompletion.create(
    model="Qwen",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True 
    # Specifying stop words in streaming output format is not yet supported and is under development.
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)

# create a request not activating streaming response
response = openai.ChatCompletion.create(
    model="Qwen",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=False,
    stop=[] # You can add custom stop words here, e.g., stop=["Observation:"] for ReAct prompting.
)
print(response.choices[0].message.content)
```

If you find `"POST /v1/chat/completions HTTP/1.1" 200 OK` in openai_api_server log, it indicates that the call was successful. 

vLLM does not support dynamic-NTK ROPE. Therefore, extending long sequences for Qwen model may lead to quality degradation(even gibberish).

### vLLM + Transformer-like Wrapper

You can download the [wrapper codes](vllm_wrapper.py) and execute the following commands for multiple rounds of dialogue interaction. (Note: It currently only supports the ``model.chat()`` method.)

```python
from vllm_wrapper import vLLMWrapper

# Bfloat16 is only supported on GPUs with compute capability of at least 8.0, 
model = vLLMWrapper('Qwen/Qwen-7B-Chat', tensor_parallel_size=1)

# run int4 model or GPUs with compute capability less than 8.0
# model = vLLMWrapper('Qwen/Qwen-7B-Chat-Int4', tensor_parallel_size=1, dtype="float16")

response, history = model.chat(query="你好", history=None)
print(response)
response, history = model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history = model.chat(query="给这个故事起一个标题", history=history)
print(response)
```
### vLLM Standalone OpenAI-like API

You can also deploy an OpenAI API server independently through vLLM. First, you need to download [chat template file](template_chatml.jinja).

Then, you can launch an OpenAI API server by following command:

```bash
python -m vllm.entrypoints.openai.api_server --model $model_path --trust-remote-code --chat-template template_chatml.jinja

# run int4 model or GPUs with compute capability less than 8.0
# python -m vllm.entrypoints.openai.api_server --model $model_path --trust-remote-code --dtype float16 --chat-template template_chatml.jinja
```

For vLLM standalone OpenAI API server, You need to set the `stop_token_ids` parameter to `[151645]` or `stop` parameter to `["<|im_end|>"]` when invoking the server.

```python
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

# create a request activating streaming response
for chunk in openai.ChatCompletion.create(
    model="Qwen",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True, 
    stop_token_ids=[151645]
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)

# create a request not activating streaming response
response = openai.ChatCompletion.create(
    model="Qwen",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=False,
    stop_token_ids=[151645]
)
print(response.choices[0].message.content)
```