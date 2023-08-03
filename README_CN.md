<br>
<p align="center">
    <img src="assets/logo.jpg" width="400"/>
<p>
<br>

<p align="center">
        Qwen-7B <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 <a> | <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>&nbsp ｜ Qwen-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 <a>| <a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>&nbsp ｜ &nbsp<a href="https://modelscope.cn/studios/qwen/Qwen-7B-Chat-Demo/summary">Demo</a>&nbsp ｜ &nbsp<a href="https://github.com/QwenLM/Qwen-7B/blob/main/tech_memo.md">Report</a>
</p>
<br>

<p align="center">
        中文</a>&nbsp ｜ &nbsp<a href="READM.md">English</a>
</p>
<br><br>

我们在🤖 **ModelScope**以及🤗 **Hugging Face**均开源了**Qwen-7B**系列模型。请在本文档顶部点击相关链接查看仓库信息。本仓库主要包括Qwen-7B的简介、使用指南、技术备忘等内容。想了解更多关于模型的信息，请点击[链接](tech_memo.md)查看我们的技术备忘录。

通义千问-7B（Qwen-7B） 是阿里云研发的通义千问大模型系列的70亿参数规模的模型。Qwen-7B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。同时，在Qwen-7B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Qwen-7B-Chat。Qwen-7B系列模型的特点包括：

1. **大规模高质量预训练数据**：我们使用了超过2.2万亿token的自建大规模预训练数据集进行语言模型的预训练。数据集包括文本和代码等多种数据类型，覆盖通用领域和专业领域。
2. **优秀的模型性能**：相比同规模的开源模型，Qwen-7B在多个评测数据集上具有显著优势，甚至超出12-13B等更大规模的模型。评测评估的能力范围包括自然语言理解与生成、数学运算解题、代码生成等。
3. **更好地支持多语言**：基于更大词表的分词器在分词上更高效，同时它对其他语言表现更加友好。用户可以在Qwen-7B的基础上更方便地训练特定语言的7B语言模型。
4. **8K的上下文长度**：Qwen-7B及Qwen-7B-Chat均能支持8K的上下文长度, 允许用户输入更长的prompt。
5. **支持插件调用**：Qwen-7B-Chat针对插件调用相关的对齐数据做了特定优化，当前模型能有效调用插件以及升级为Agent。

## 新闻

* 2023年8月3日 在魔搭社区（ModelScope）和Hugging Face同步推出Qwen-7B和Qwen-7B-Chat模型。同时，我们发布了技术备忘录，介绍了相关的训练细节和模型表现。

## 评测表现

Qwen-7B在多个全面评估自然语言理解与生成、数学运算解题、代码生成等能力的评测数据集上，包括MMLU、C-Eval、GSM8K、HumanEval、WMT22等，均超出了同规模大语言模型的表现，甚至超出了如12-13B参数等更大规模的语言模型。

| Model        | MMLU     |   C-Eval |    GSM8K | HumanEval | WMT22 (en-zh) |
| :------------- | ---------- | ---------: | ---------: | ----------: | --------------: |
| LLaMA-7B     | 35.1     |        - |     11.0 |      10.5 |           8.7 |
| LLaMA 2-7B   | 45.3     |        - |     14.6 |      12.8 |          17.9 |
| Baichuan-7B  | 42.3     |     42.8 |      9.7 |       9.2 |          26.6 |
| ChatGLM2-6B  | 47.9     |     51.7 |     32.4 |       9.2 |             - |
| InternLM-7B  | 51.0     |     52.8 |     31.2 |      10.4 |          14.8 |
| Baichuan-13B | 51.6     |     53.6 |     26.6 |      12.8 |          30.0 |
| LLaMA-13B    | 46.9     |     35.5 |     17.8 |      15.8 |          12.0 |
| LLaMA 2-13B  | 54.8     |        - |     28.7 |      18.3 |          24.2 |
| ChatGLM2-12B | 56.2     | **61.6** |     40.9 |         - |             - |
| **Qwen-7B**  | **56.7** |     59.6 | **51.6** |  **24.4** |      **30.6** |

<p align="center">
    <img src="assets/performance.png" width="1000"/>
<p>
<br>

更多的实验结果和细节请查看我们的技术备忘录。点击[这里](techmemo-draft.md)。

## 快速使用

我们提供简单的示例来说明如何利用🤖 ModelScope和🤗 Transformers快速使用Qwen-7B和Qwen-7B-Chat。

在开始前，请确保你已经配置好环境并安装好相关的代码包。最重要的是，确保你的pytorch版本高于`1.12`，然后安装相关的依赖库。

```bash
pip install transformers==4.31.0 accelerate tiktoken einops
```

我们还推荐安装[flash-attention](https://github.com/Dao-AILab/flash-attention)来提高你的运行效率以及降低显存占用。

```bash
git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
pip install csrc/layer_norm
pip install csrc/rotary
```

接下来你可以开始使用Transformers或者ModelScope来使用我们的模型。

#### 🤗 Transformers

如希望使用Qwen-7B-chat进行推理，所需要写的只是如下所示的数行代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
## 使用bf16精度
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
## 使用fp16精度
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 默认使用fp32精度
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

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
## 使用bf16精度
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, bf16=True).eval()
## 使用fp16精度
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 默认使用fp32精度
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
inputs = inputs.to('cuda:0')
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

## 量化

如希望使用更低精度的量化模型，如4比特和8比特的模型，我们提供了简单的示例来说明如何快速使用量化模型。在开始前，确保你已经安装了`bitsandbytes`。

```bash
pip install bitsandbytes
```

你只需要在`AutoModelForCausalLM.from_pretrained`中添加你的量化配置，即可使用量化模型。如下所示：

```python
from transformers import BitsAndBytesConfig

# quantization configuration for NF4 (4 bits)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

# quantization configuration for Int8 (8 bits)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    args.checkpoint_path,
    device_map="cuda:0",
    quantization_config=quantization_config,
    max_memory=max_memory,
    trust_remote_code=True,
).eval()
```

上述方法可以让我们将模型量化成`NF4`和`Int8`精度的模型进行读取，帮助我们节省显存开销。我们也提供了相关性能数据。我们发现尽管模型在效果上存在损失，但模型的显存开销大幅降低。

| Precision | MMLU | Memory |
| :---------: | -------: | -----: |
|   BF16   |  56.7 |   16.2G |
|   Int8   |  52.8 |   10.1G |
|    NF4    |  48.9 |    7.4G |

## 工具调用

Qwen-7B-Chat针对包括API、数据库、模型等工具在内的调用进行了优化。用户可以开发基于Qwen-7B的LangChain、Agent甚至Code Interpreter。我们在内部的即将开源的评测数据集上测试模型的工具调用能力，并发现Qwen-7B-Chat能够取得稳定的表现。

| Model       | Tool Selection (Acc.↑) | Tool Input (Rouge-L↑) | False Positive Error↓ |
| ------------- | ------------------------- | ------------------------ | ------------------------ |
| GPT-4       | 95%                     | **0.90**               | 15%                    |
| GPT-3.5     | 85%                     | 0.88                   | 75%                    |
| **Qwen-7B** | **99%**                 | 0.89                   | **8.5%**               |

我们提供了文档说明如何根据ReAct Prompting的原则写作你的prompt。

For how to write and use prompts for ReAct Prompting, please refer to [the ReAct examples](examples/react_prompt.md)。

此外，我们还提供了实验结果表明我们的模型扮演Agent的能力。请阅读相关文档[链接](https://huggingface.co/docs/transformers/transformers_agents)了解更多信息。模型在Hugging Face提供的评测数据集上表现如下：

| Model           | Tool Selection↑ | Tool Used↑ | Code↑    |
| ----------------- | ------------------ | ------------- | ----------- |
| GPT-4           | **100**          | **100**     | **97.41** |
| GPT-3.5         | 95.37            | 96.30       | 87.04     |
| StarCoder-15.5B | 87.04            | 87.96       | 68.89     |
| **Qwen-7B**     | 90.74            | 92.59       | 74.07     |

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

## 使用协议

研究人员与开发者可使用Qwen-7B和Qwen-7B-Chat或进行二次开发。我们同样允许商业使用，具体细节请查看[LICENSE](LICENSE)。

## 联系我们

如果你想给我们的研发团队和产品团队留言，请通过邮件（qianwen_opensource@alibabacloud.com）联系我们。

