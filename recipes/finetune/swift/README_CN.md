## 介绍
[SWIFT](https://github.com/modelscope/swift)（Scalable lightWeight Infrastructure for Fine-Tuning）是一个可扩展的轻量级一站式训练、推理深度学习框架。它集成了各种高效的微调方法，如LoRA、QLoRA、阿里云自研的ResTuning-Bypass等，以及开箱即用的训练推理脚本，使开发者可以在单张商业级显卡上微调推理LLM&AIGC模型。此外，SWIFT与PEFT完全兼容，使开发者可以在ModelScope模型体系中使用PEFT的能力。

## 安装
```shell
# 设置pip全局镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]

# 如果你想要使用deepspeed.
pip install deepspeed -U

# 如果你想要使用基于auto_gptq的qlora训练. (推荐, 效果优于bnb)
# 支持auto_gptq的模型: `https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md#模型`
# auto_gptq和cuda版本有对应关系，请按照`https://github.com/PanQiWei/AutoGPTQ#quick-installation`选择版本
pip install auto_gptq -U

# 如果你想要使用基于bnb的qlora训练.
pip install bitsandbytes -U

# 环境对齐 (如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```


## webui使用

执行如下命令启动webui通过界面方式进行模型训练推理
```shell
swift web-ui
```
界面示例如下
![image](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/swift_webui.jpg)

## 微调
```python
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \

# 使用自己的数据集
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --custom_train_dataset_path chatml.jsonl \
    --output_dir output \

# 使用DDP
# Experimental environment: 2 * 3090
# 2 * 23GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \

# 多机多卡
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
更多微调方法参考[这里](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.md#%E5%BE%AE%E8%B0%83)

已有微调代码示例
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

## 推理

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
# kwargs['use_flash_attn'] = True  # 使用flash_attn

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)
# 修改max_new_tokens
model.generation_config.max_new_tokens = 128

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '浙江的省会在哪里？'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = '这有什么好吃的？'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: 浙江的省会在哪里？
response: 浙江省的省会是杭州。
query: 这有什么好吃的？
response: 杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。
history: [('浙江的省会在哪里？', '浙江省的省会是杭州。'), ('这有什么好吃的？', '杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。')]
"""

# 流式输出对话模板
inference(model, template, '第一个问题是什么', history, verbose=True, stream=True)
"""Out[1]
[PROMPT]<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
浙江的省会在哪里？<|im_end|>
<|im_start|>assistant
浙江省的省会是杭州。<|im_end|>
<|im_start|>user
这有什么好吃的？<|im_end|>
<|im_start|>assistant
杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。<|im_end|>
<|im_start|>user
第一个问题是什么<|im_end|>
<|im_start|>assistant
[OUTPUT]你的第一个问题是“浙江的省会在哪里？”<|im_end|>
"""
```
更多推理使用请参考[这里](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E6%8E%A8%E7%90%86%E6%96%87%E6%A1%A3.md)
