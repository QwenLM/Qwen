## Introduction
[SWIFT](https://github.com/modelscope/swift) (Scalable lightWeight Infrastructure for Fine-Tuning) is an extensible framwork designed to faciliate lightweight model fine-tuning and inference. It integrates implementations for various efficient fine-tuning methods, by embracing approaches that is parameter-efficient, memory-efficient, and time-efficient. SWIFT integrates seamlessly into ModelScope ecosystem and offers the capabilities to finetune various models, with a primary emphasis on LLMs and vision models. Additionally, SWIFT is fully compatible with PEFT, enabling users to leverage the familiar Peft interface to finetune ModelScope models.

## Installation

```shell
# Set the global pip mirror
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]

# If you want to use deepspeed
pip install deepspeed -U

# If you want to use qlora training based on auto_gptq (recommended, performs better than bnb)
# Models supporting auto_gptq: `https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md#模型`
# There's a version correspondence between auto_gptq and cuda; refer to `https://github.com/PanQiWei/AutoGPTQ#quick-installation` for selecting the appropriate version
pip install auto_gptq -U

# If you want to use qlora training based on bnb
pip install bitsandbytes -U

# Environment alignment (run the following commands if you encounter errors; the repository is tested with the latest environment)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## WebUI Usage

Run the following command to start the webui and conduct model training and inference through the graphical interface:
```shell
swift web-ui
```
A screenshot example can be found at:
![image](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/swift_webui.jpg)

## Fine-tuning

```python
# Experimental environment: A10, 3090, V100, ...
# GPU memory requirement: 20GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \

# Use your own dataset
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --custom_train_dataset_path chatml.jsonl \
    --output_dir output \

# Using DDP (Distributed Data Parallel)
# Experimental environment: 2 * 3090
# GPU memory requirement: 2 * 23GB
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \

# Multi-machine multi-GPU setup
# node0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \
# node1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \
```
For more fine-tuning methods, please refer to [here](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.md#%E5%BE%AE%E8%B0%83).



Examples

| 模型名称          | 训练方法                                                                                             |
|:-------------------|:---------------------------------------------------------------------------------------------------------------------------|
| qwen_14b           | [lora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b/lora_ddp_ds)             |
| qwen_14b           | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b/qlora)                         |
| qwen_14b           | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b/qlora_ddp_ds)           |
| qwen_14b_chat      | [lora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/lora_ddp_ds)        |
| qwen_14b_chat      | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/qlora)                    |
| qwen_14b_chat      | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/qlora_ddp_ds)      |
| qwen_14b_chat_int4 | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat_int4/qlora)               |
| qwen_14b_chat_int4 | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat_int4/qlora_ddp_ds) |
| qwen_14b_chat_int8 | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat_int8/qlora)               |
| qwen_14b_chat_int8 | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat_int8/qlora_ddp_ds) |
| qwen_1_8b_chat     | [full](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_1_8b_chat/full)                     |
| qwen_1_8b_chat     | [full_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_1_8b_chat/full_ddp)             |
| qwen_72b_chat      | [lora_mp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/lora_mp)                |
| qwen_72b_chat      | [lora_mp_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/lora_mp_ddp)        |
| qwen_72b_chat      | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/qlora)                    |
| qwen_72b_chat_int4 | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat_int4/qlora_ddp_ds) |
| qwen_72b_chat_int8 | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat_int8/qlora_ddp_ds) |
| qwen_7b            | [lora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b/lora_ddp_ds)              |
| qwen_7b            | [qlora_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b/qlora_ddp)                  |
| qwen_7b_chat       | [full](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full)                       |
| qwen_7b_chat       | [full_freeze_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_freeze_ddp) |
| qwen_7b_chat       | [full_mp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_mp)                 |
| qwen_7b_chat       | [full_mp_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_mp_ddp)         |
| qwen_7b_chat       | [lora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/lora)                       |
| qwen_7b_chat       | [lora_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/lora_ddp)               |
| qwen_7b_chat       | [lora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/lora_ddp_ds)         |
| qwen_7b_chat       | [lora_mp_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/lora_mp_ddp)         |
| qwen_7b_chat       | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/qlora)                     |
| qwen_7b_chat       | [qlora_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/qlora_ddp)             |
| qwen_7b_chat       | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/qlora_ddp_ds)       |
| qwen_7b_chat_int4  | [qalora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat_int4/qalora)              |
| qwen_7b_chat_int4  | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat_int4/qlora)                |
| qwen_7b_chat_int4  | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat_int4/qlora_ddp_ds)  |
| qwen_7b_chat_int8  | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat_int8/qlora)                |
| qwen_7b_chat_int8  | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat_int8/qlora_ddp_ds)  |
| qwen_audio_chat    | [full_mp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_audio_chat/full_mp)              |
| qwen_audio_chat    | [full_mp_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_audio_chat/full_mp_ddp)      |
| qwen_audio_chat    | [lora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_audio_chat/lora)                    |
| qwen_audio_chat    | [lora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_audio_chat/lora_ddp_ds)      |
| qwen_vl            | [lora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_vl/lora_ddp_ds)              |
| qwen_vl_chat       | [full_mp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_vl_chat/full_mp)                 |
| qwen_vl_chat       | [full_mp_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_vl_chat/full_mp_ddp)         |
| qwen_vl_chat       | [lora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_vl_chat/lora)                       |
| qwen_vl_chat       | [lora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_vl_chat/lora_ddp_ds)         |
| qwen_vl_chat       | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_vl_chat/qlora)                     |
| qwen_vl_chat_int4  | [qlora](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_vl_chat_int4/qlora)                |
| qwen_vl_chat_int4  | [qlora_ddp_ds](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_vl_chat_int4/qlora_ddp_ds)  |


## Inference

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen


kwargs = {}
# kwargs['use_flash_attn'] = True  # Use flash_attn if desired

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)
# Modify max_new_tokens
model.generation_config.max_new_tokens = 128

template = get_template(template_type, tokenizer)
seed_everything(42)
query = 'What is the provincial capital of Zhejiang?'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

query = 'What delicious food can be found here?'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Output[0]:
query: What is the provincial capital of Zhejiang?
response: The provincial capital of Zhejiang is Hangzhou.
query: What delicious food can be found here?
response: Hangzhou has many famous delicacies, such as West Lake Vinegar Fish, Longjing Shrimp, Sweet and Sour Spare Ribs, and Maodu. Additionally, there are unique Hangzhou-style pastries like Osmanthus Cake, Lotus Paste Pastry, and Aiwo Steamed Rice Cakes.
history: [('What is the provincial capital of Zhejiang?', 'The provincial capital of Zhejiang is Hangzhou.'), ('What delicious food can be found here?', 'Hangzhou has many famous delicacies, such as West Lake Vinegar Fish, Longjing Shrimp, Sweet and Sour Spare Ribs, and Maodu. Additionally, there are unique Hangzhou-style pastries like Osmanthus Cake, Lotus Paste Pastry, and Aiwo Steamed Rice Cakes.')]
"""

# Streaming dialogue output with verbose mode
inference(model, template, 'What was the first question?', history, verbose=True, stream=True)
"""Output[1]:
[PROMPT]
You asked your first question, "What is the provincial capital of Zhejiang?"
[OUTPUT] Your first question was “What is the provincial capital of Zhejiang?”
"""

For more on inference usage, please refer to [here](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM_Inference_Guide.md).
