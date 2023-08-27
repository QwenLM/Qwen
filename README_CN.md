<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp ｜ &nbsp<a href="README_JA.md">日本語</a>
</p>
<br><br>

<p align="center">
    <img src="assets/logo.jpg" width="400"/>
<p>
<br>

<p align="center">
        Qwen-7B <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 <a> | <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>&nbsp ｜ Qwen-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 <a> | <a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>&nbsp | Qwen-7B-Chat-Int4 <a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int4">🤗</a>
<br>
<a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/qwen_wechat_group.PNG">WeChat</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/z3GAxXZ9Ce">Discord</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://modelscope.cn/studios/qwen/Qwen-7B-Chat-Demo/summary">Demo</a>&nbsp ｜ &nbsp<a href="https://github.com/QwenLM/Qwen-7B/blob/main/tech_memo.md">Report</a>
</p>
<br><br>



我们在🤖 **ModelScope**以及🤗 **Hugging Face**均开源了**Qwen-7B**系列模型。请在本文档顶部点击相关链接查看仓库信息。本仓库主要包括Qwen-7B的简介、使用指南、技术备忘等内容。想了解更多关于模型的信息，请点击[链接](tech_memo.md)查看我们的技术备忘录。

通义千问-7B（Qwen-7B） 是阿里云研发的通义千问大模型系列的70亿参数规模的模型。Qwen-7B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。同时，在Qwen-7B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Qwen-7B-Chat。Qwen-7B系列模型的特点包括：

1. **大规模高质量预训练数据**：我们使用了超过2.2万亿token的自建大规模预训练数据集进行语言模型的预训练。数据集包括文本和代码等多种数据类型，覆盖通用领域和专业领域。
2. **优秀的模型性能**：相比同规模的开源模型，Qwen-7B在多个评测数据集上具有显著优势，甚至超出12-13B等更大规模的模型。评测评估的能力范围包括自然语言理解与生成、数学运算解题、代码生成等。
3. **更好地支持多语言**：基于更大词表的分词器在分词上更高效，同时它对其他语言表现更加友好。用户可以在Qwen-7B的基础上更方便地训练特定语言的7B语言模型。
4. **8K的上下文长度**：Qwen-7B及Qwen-7B-Chat均能支持8K的上下文长度, 允许用户输入更长的prompt。
5. **支持插件调用**：Qwen-7B-Chat针对插件调用相关的对齐数据做了特定优化，当前模型能有效调用插件以及升级为Agent。

以下章节的信息可能对你有帮助，建议阅读。如果你在使用过程遇到问题，建议先查询FAQ，如仍无法解决再提交issue。

## 新闻

* 2023年8月21日 发布Qwen-7B-Chat的Int4量化模型，Qwen-7B-Chat-Int4。该模型显存占用低，推理速度相比半精度模型显著提升，在基准评测上效果损失较小。
* 2023年8月3日 在魔搭社区（ModelScope）和Hugging Face同步推出Qwen-7B和Qwen-7B-Chat模型。同时，我们发布了技术备忘录，介绍了相关的训练细节和模型表现。

## 评测表现

Qwen-7B在多个全面评估自然语言理解与生成、数学运算解题、代码生成等能力的评测数据集上，包括MMLU、C-Eval、GSM8K、HumanEval、WMT22、CMMLU等，均超出了同规模大语言模型的表现，甚至超出了如12-13B参数等更大规模的语言模型。

| Model             | MMLU           |         C-Eval |          GSM8K |      HumanEval |  WMT22 (en-zh) |         CMMLU |
| :---------------- | :------------: | :------------: | :------------: | :------------: | :------------: |:------------: |
| LLaMA-7B          | 35.1           |              - |           11.0 |           10.5 |            8.7 |             - |
| LLaMA 2-7B        | 45.3           |              - |           14.6 |           12.8 |           17.9 |             - |
| Baichuan-7B       | 42.3           |           42.8 |            9.7 |            9.2 |           26.6 |          44.4 |
| ChatGLM2-6B       | 47.9           |           51.7 |           32.4 |            9.2 |              - |          48.8 |
| InternLM-7B       | 51.0           |           52.8 |           31.2 |           10.4 |           14.8 |             - |
| Baichuan-13B      | 51.6           |           53.6 |           26.6 |           12.8 |           30.0 |          55.8 |
| LLaMA-13B         | 46.9           |           35.5 |           17.8 |           15.8 |           12.0 |             - |
| LLaMA 2-13B       | 54.8           |              - |           28.7 |           18.3 |           24.2 |             - |
| ChatGLM2-12B      | 56.2           |       **61.6** |           40.9 |              - |              - |             - |
| **Qwen-7B**       | **56.7**       |           59.6 |       **51.6** |       **24.4** |       **30.6** |      **58.8** |

<p align="center">
    <img src="assets/performance.png" width="1000"/>
<p>
<br>

此外，根据[OpenCompass](https://opencompass.org.cn/leaderboard-llm)进行的大型语言模型第三方评估，Qwen-7B 和 Qwen-7B-Chat 是其中表现最优的7B参数模型。该评估由大量公开基准组成，用于评估语言理解和生成、代码生成、数学、推理等。

更多的实验结果和细节请查看我们的技术备忘录。点击[这里](tech_memo.md)。

## 要求

* python 3.8及以上版本
* pytorch 1.12及以上版本，推荐2.0及以上版本
* 建议使用CUDA 11.4及以上（GPU用户、flash-attention用户等需考虑此选项）

## 快速使用

我们提供简单的示例来说明如何利用🤖 ModelScope和🤗 Transformers快速使用Qwen-7B和Qwen-7B-Chat。

在开始前，请确保你已经配置好环境并安装好相关的代码包。最重要的是，确保你满足上述要求，然后安装相关的依赖库。

```bash
pip install -r requirements.txt
```

如果你的显卡支持fp16或bf16精度，我们还推荐安装[flash-attention](https://github.com/Dao-AILab/flash-attention)来提高你的运行效率以及降低显存占用。(**flash-attention只是可选项，不安装也可正常运行该项目**)

```bash
git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 下方安装可选，安装可能比较缓慢。
# pip install csrc/layer_norm
# pip install csrc/rotary
```

接下来你可以开始使用Transformers或者ModelScope来使用我们的模型。

#### 🤗 Transformers

如希望使用Qwen-7B-chat进行推理，所需要写的只是如下所示的数行代码。**请确保你使用的是最新代码。**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认使用自动模式，根据设备自动选择精度
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# 第一轮对话 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 第二轮对话 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# 第三轮对话 3rd dialogue turn
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》
```

运行Qwen-7B同样非常简单。

<details>
  <summary>运行Qwen-7B</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="cpu", trust_remote_code=True).eval()
# 默认使用自动模式，根据设备自动选择精度
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# 蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是亚的斯亚贝巴（Addis Ababa）...
```

</details>

#### 🤖 ModelScope

魔搭（ModelScope）是开源的模型即服务共享平台，为泛AI开发者提供灵活、易用、低成本的一站式模型服务产品。使用ModelScope同样非常简单，代码如下所示：

```python
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download

model_id = 'QWen/qwen-7b-chat'
revision = 'v1.0.0'

model_dir = snapshot_download(model_id, revision)

pipe = pipeline(
task=Tasks.chat, model=model_dir, device_map='auto')
history = None

text = '浙江的省会在哪里？'
results = pipe(text, history=history)
response, history = results['response'], results['history']
print(f'Response: {response}')
text = '它有什么好玩的地方呢？'
results = pipe(text, history=history)
response, history = results['response'], results['history']
print(f'Response: {response}')
```

## Tokenization

> 注：作为术语的“tokenization”在中文中尚无共识的概念对应，本文档采用英文表达以利说明。

基于tiktoken的tokenizer有别于其他分词器，比如sentencepiece tokenizer。尤其在微调阶段，需要特别注意特殊token的使用。关于tokenizer的更多信息，以及微调时涉及的相关使用，请参阅[文档](tokenization_note_zh.md)。

## 量化

### 用法

**请注意：我们更新量化方案为基于[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)的量化，提供Qwen-7B-Chat的Int4量化模型[点击这里](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4)。相比此前方案，该方案在模型评测效果几乎无损，且存储需求更低，推理速度更优。**

以下我们提供示例说明如何使用Int4量化模型。在开始使用前，请先保证满足AutoGPTQ的要求，并使用源代码安装（由于最新支持Qwen的代码未发布到PyPI）：

```bash
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install .
```

随后便能轻松读取量化模型：

```python
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("Qwen/Qwen-7B-Chat-Int4", device_map="auto", trust_remote_code=True, use_safetensors=True).eval()
```

推理方法和基础用法类似，但注意需要从外部传入generation config：

```python
from transformers import GenerationConfig
config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)
response, history = model.chat(tokenizer, "Hi", history=None, generation_config=config)
```

### 效果评测

我们对BF16和Int4模型在基准评测上做了测试，发现量化模型效果损失较小，结果如下所示：

|  Quantization |   MMLU     |  CEval (val) |  GSM8K |  Humaneval |
| ------------- | :--------: | :----------: | :----: | :--------: |
| BF16          |    53.9    |     54.2     |  41.1  |    24.4    |
| Int4          |    52.6    |     52.9     |  38.1  |    23.8    |

### 推理速度

我们测算了BF16和Int4模型生成2048和8192个token的平均推理速度（tokens/s）。如图所示：

|  Quantization | Speed (2048 tokens) | Speed (8192 tokens) |
| ------------- | :------------------:| :------------------:|
|      BF16     | 30.53               | 28.51               |
|      Int4     | 45.60               | 33.83               |

具体而言，我们记录在长度为1的上下文的条件下生成8192个token的性能。评测运行于单张A100-SXM4-80G GPU，使用PyTorch 2.0.1和CUDA 11.4。推理速度是生成8192个token的速度均值。

### 显存使用

我们还测算了BF16和Int4模型编码2048个token及生成8192个token的峰值显存占用情况。结果如下所示：

| Quantization Level | Peak Usage for Encoding 2048 Tokens | Peak Usage for Generating 8192 Tokens |
| ------------------ | :---------------------------------: | :-----------------------------------: |
| BF16               |               18.99GB               |                24.40GB                |
| Int4               |               10.20GB                |                15.61GB                |

上述性能测算使用[此脚本](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py)完成。

## Demo

### Web UI

我们提供了Web UI的demo供用户使用 (感谢 @wysaid 支持)。在开始前，确保已经安装如下代码库：

```
pip install -r requirements_web_demo.txt
```

随后运行如下命令，并点击生成链接：

```
python web_demo.py
```

<p align="center">
    <br>
    <img src="assets/web_demo.gif" width="600" />
    <br>
<p>

### 交互式Demo

我们提供了一个简单的交互式Demo示例，请查看`cli_demo.py`。当前模型已经支持流式输出，用户可通过输入文字的方式和Qwen-7B-Chat交互，模型将流式输出返回结果。运行如下命令：

```
python cli_demo.py
```

<p align="center">
    <br>
    <img src="assets/cli_demo.gif" width="600" />
    <br>
<p>

## API

我们提供了OpenAI API格式的本地API部署方法（感谢@hanpenggit）。在开始之前先安装必要的代码库：

```bash
pip install fastapi uvicorn openai pydantic sse_starlette
```

随后即可运行以下命令部署你的本地API：

```bash
python openai_api.py
```

你也可以修改参数，比如`-c`来修改模型名称或路径, `--cpu-only`改为CPU部署等等。如果部署出现问题，更新上述代码库往往可以解决大多数问题。

使用API同样非常简单，示例如下：

```python
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

# 使用流式回复的请求
for chunk in openai.ChatCompletion.create(
    model="Qwen-7B",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
    # 流式输出的自定义stopwords功能尚未支持，正在开发中
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)

# 不使用流式回复的请求
response = openai.ChatCompletion.create(
    model="Qwen-7B",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=False,
    stop=[] # 在此处添加自定义的stop words 例如ReAct prompting时需要增加： stop=["Observation:"]。
)
print(response.choices[0].message.content)
```

<p align="center">
    <br>
    <img src="assets/openai_api.gif" width="600" />
    <br>
<p>

## 工具调用

Qwen-7B-Chat针对包括API、数据库、模型等工具在内的调用进行了优化。用户可以开发基于Qwen-7B的LangChain、Agent甚至Code Interpreter。在我们开源的[评测数据集](eval/EVALUATION.md)上测试模型的工具调用能力，并发现Qwen-7B-Chat能够取得稳定的表现。

| Model       | Tool Selection (Acc.↑) | Tool Input (Rouge-L↑)  | False Positive Error↓  |
|:------------|:----------------------:|:----------------------:|:----------------------:|
| GPT-4       | 95%                    | **0.90**               | 15%                    |
| GPT-3.5     | 85%                    | 0.88                   | 75%                    |
| **Qwen-7B** | **99%**                | 0.89                   | **9.7%**               |

我们提供了文档说明如何根据ReAct Prompting的原则写作你的prompt。

For how to write and use prompts for ReAct Prompting, please refer to [the ReAct examples](examples/react_prompt.md)。

此外，我们还提供了实验结果表明我们的模型扮演Agent的能力。请阅读相关文档[链接](https://huggingface.co/docs/transformers/transformers_agents)了解更多信息。模型在Hugging Face提供的评测数据集上表现如下：

| Model          | Tool Selection↑ | Tool Used↑  |   Code↑   |
|:---------------|:---------------:|:-----------:|:---------:|
|GPT-4           |     **100**     |   **100**   | **97.41** |
|GPT-3.5         |      95.37      |    96.30    |   87.04   |
|StarCoder-15.5B |      87.04      |    87.96    |   68.89   |
| **Qwen-7B**    |      90.74      |    92.59    |   74.07   |

## 长文本理解

我们引入了NTK插值、窗口注意力、LogN注意力缩放等技术来提升模型的上下文长度并突破训练序列长度的限制。我们的模型已经突破8K的序列长度。通过arXiv数据集上的语言模型实验，我们发现Qwen-7B能够在长序列的设置下取得不错的表现。

<table>
    <tr>
        <th rowspan="2">Model</th><th colspan="5" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">1024</th><th align="center">2048</th><th align="center">4096</th><th align="center">8192</th><th align="center">16384</th>
    </tr>
    <tr>
        <td>Qwen-7B</td><td align="center"><b>4.23</b></td><td align="center"><b>3.78</b></td><td align="center">39.35</td><td align="center">469.81</td><td align="center">2645.09</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk</td><td align="center"><b>4.23</b></td><td align="center"><b>3.78</b></td><td align="center">3.59</td><td align="center">3.66</td><td align="center">5.71</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn</td><td align="center"><b>4.23</b></td><td align="center"><b>3.78</b></td><td align="center"><b>3.58</b></td><td align="center">3.56</td><td align="center">4.62</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn + local_attn</td><td align="center"><b>4.23</b></td><td align="center"><b>3.78</b></td><td align="center"><b>3.58</b></td><td align="center"><b>3.49</b></td><td align="center"><b>4.32</b></td>
    </tr>
</table>

## 复现

我们提供了评测脚本以供复现我们的实验结果。注意，由于内部代码和开源代码存在少许差异，评测结果可能与汇报结果存在细微的结果不一致。请阅读[eval/EVALUATION.md](eval/EVALUATION.md)了解更多信息。

## FAQ

如遇到问题，敬请查阅[FAQ](FAQ_zh.md)以及issue区，如仍无法解决再提交issue。

## 使用协议

研究人员与开发者可使用Qwen-7B和Qwen-7B-Chat或进行二次开发。我们同样允许商业使用，具体细节请查看[LICENSE](LICENSE)。如需商用，请填写[问卷](https://dashscope.console.aliyun.com/openModelApply/qianwen)申请。

## 联系我们

如果你想给我们的研发团队和产品团队留言，请通过邮件（qianwen_opensource@alibabacloud.com）联系我们。

