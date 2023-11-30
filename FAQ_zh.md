# FAQ

## 安装&环境

#### flash attention 安装失败

flash attention是一个用于加速模型训练推理的可选项，且仅适用于Turing、Ampere、Ada、Hopper架构的Nvidia GPU显卡（如H100、A100、RTX 3090、T4、RTX 2080），您可以在不安装flash attention的情况下正常使用模型进行推理。

#### 我应该用哪个transformers版本？

建议使用4.32.0。

#### 我把模型和代码下到本地，按照教程无法使用，该怎么办？

答：别着急，先检查你的代码是不是更新到最新版本，然后确认你是否完整地将模型checkpoint下到本地。

#### `qwen.tiktoken`这个文件找不到，怎么办？

这个是我们的tokenizer的merge文件，你必须下载它才能使用我们的tokenizer。注意，如果你使用git clone却没有使用git-lfs，这个文件不会被下载。如果你不了解git-lfs，可点击[官网](https://git-lfs.com/)了解。

#### transformers_stream_generator/tiktoken/accelerate，这几个库提示找不到，怎么办？

运行如下命令：`pip install -r requirements.txt`。相关依赖库在[https://github.com/QwenLM/Qwen-7B/blob/main/requirements.txt](https://github.com/QwenLM/Qwen/blob/main/requirements.txt) 可以找到。
<br><br>


## Demo & 推理

#### 是否提供Demo？CLI Demo及Web UI Demo？

`web_demo.py`和`cli_demo.py`分别提供了Web UI以及CLI的Demo。请查看README相关内容了解更多。

#### 我没有GPU，只用CPU运行CLI demo可以吗？

可以的，运行`python  cli_demo.py --cpu-only`命令即可将模型读取到CPU并使用CPU进行推理。

#### Qwen支持流式推理吗？

Qwen当前支持流式推理。见位于`modeling_qwen.py`的`chat_stream`函数。

#### 使用`chat_stream()`生成混乱的内容及乱码，为什么？

这是由于模型生成过程中输出的部分token需要与后续token一起解码才能输出正常文本，单个token解码结果是无意义字符串，我们已经更新了tokenizer解码时的默认设置，避免这些字符串在生成结果中出现，如果仍有类似问题请更新模型至最新版本。

#### 模型的输出看起来与输入无关/没有遵循指令/看起来呆呆的

请检查是否加载的是Qwen-Chat模型进行推理，Qwen模型是未经align的预训练基模型，不期望具备响应用户指令的能力。我们在模型最新版本已经对`chat`及`chat_stream`接口内进行了检查，避免您误将预训练模型作为SFT/Chat模型使用。

#### 是否有量化版本模型

目前Qwen支持基于AutoGPTQ的4-bit的量化推理。

#### 生成序列较长后速度显著变慢

请更新到最新代码。

#### 处理长序列时效果有问题

请确认是否开启ntk。若要启用这些技巧，请将`config.json`里的`use_dynamc_ntk`和`use_logn_attn`设置为`true`。最新代码默认为`true`。
<br><br>


## 微调

#### 当前是否支持SFT和RLHF？

我们目前提供了SFT的代码，支持全参数微调、LoRA和Q-LoRA。此外，当前有多个外部项目也已实现支持，如[FastChat](**[https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat))、[Firefly]([https://github.com/yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly))、[**LLaMA Efficient Tuning**]([https://github.com/hiyouga/LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning))等。我们会尽快更新这部分代码和说明。

我们还没提供对RLHF训练的支持，敬请期待。
<br><br>


## Tokenizer

#### bos_id/eos_id/pad_id，这些token id不存在，为什么？

在训练过程中，我们仅使用<|endoftext|>这一token作为sample/document之间的分隔符及padding位置占位符，你可以将bos_id, eos_id, pad_id均指向tokenizer.eod_id。请阅读我们关于tokenizer的文档，了解如何设置这些id。


## Docker

#### 下载官方Docker镜像速度很慢

在下载官方镜像时，您可能由于某些网络原因导致下载速度变慢。可以参考[阿里云容器镜像服务](https://help.aliyun.com/zh/acr/user-guide/accelerate-the-pulls-of-docker-official-images)加速官方镜像的下载。