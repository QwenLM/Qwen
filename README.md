<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp ｜ &nbsp<a href="README_JA.md">日本語</a> ｜ &nbsp<a href="README_FR.md">Français</a> ｜ &nbsp<a href="README_ES.md">Español</a>
</p>
<br><br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg" width="400"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2309.16609">Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-72B-Chat-Demo/summary">Demo</a>
<br>
<a href="assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/zmemtgyAxT">Discord</a>&nbsp&nbsp ｜  &nbsp&nbsp<a href="https://dashscope.aliyun.com">API</a> 
</p>
<br><br>

|     |                                                              Qwen-Chat                                                               |                                                                Qwen-Chat (Int4)                                                                |                        Qwen-Chat (Int8)                         |                                                            Qwen                                                            |
|-----|:------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|
| 1.8B  |  <a href="https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-1_8B-Chat">🤗</a>  |  <a href="https://modelscope.cn/models/qwen/Qwen-1_8B-Chat-Int4/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-1_8B-Chat-Int4">🤗</a>  | <a href="https://modelscope.cn/models/qwen/Qwen-1_8B-Chat-Int8/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-1_8B-Chat-Int8">🤗</a>  |  <a href="https://modelscope.cn/models/qwen/Qwen-1_8B/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-1_8B">🤗</a>  |
| 7B  |  <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>  |  <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int4/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int4">🤗</a>  | <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int8">🤗</a>  |  <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>  |
| 14B | <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-14B-Chat">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int4/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-14B-Chat-Int4">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int8/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-14B-Chat-Int8">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-14B/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-14B">🤗</a> |
| 72B | <a href="https://modelscope.cn/models/qwen/Qwen-72B-Chat/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-72B-Chat">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-72B-Chat-Int4/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-72B-Chat-Int4">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-72B-Chat-Int8/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-72B-Chat-Int8">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-72B/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-72B">🤗</a> |



We opensource our **Qwen** series, now including **Qwen**, the base language models, namely **Qwen-1.8B**, **Qwen-7B**, **Qwen-14B**, and **Qwen-72B**, as well as **Qwen-Chat**, the chat models, namely **Qwen-1.8B-Chat**, **Qwen-7B-Chat**, **Qwen-14B-Chat**, and **Qwen-72B-Chat**. Links are on the above table. Click them and check the model cards. Also, we release the **[technical report](https://arxiv.org/abs/2309.16609)**. Please click the paper link and check it out!

In brief, we have strong base language models, which have been stably pretrained for up to 3 trillion tokens of multilingual data with a wide coverage of domains, languages (with a focus on Chinese and English), etc. They are able to achieve competitive performance on benchmark datasets. Additionally, we have chat models that are aligned with human preference based on SFT and RLHF (not released yet), which are able to chat, create content, extract information, summarize, translate, code, solve math problems, and so on, and are able to use tools, play as agents, or even play as code interpreters, etc.

| Model     | Release Date | Max Length | System Prompt Enhancement | # of Pretrained Tokens | Minimum GPU Memory Usage of Finetuning (Q-Lora) | Minimum GPU Usage of Generating 2048 Tokens (Int4) | Tool Usage |
|:----------|:------------:|:----------:|:-------------------------:|:----------------------:|:-----------------------------------------------:|:--------------------------------------------------:|:----------:|
| Qwen-1.8B |   23.11.30   |    32K     |             ✅             |          2.2T          |                      5.8GB                      |                       2.9GB                        |     ✅      |  
| Qwen-7B   |   23.08.03   |    32K     |             ❎             |          2.4T          |                     11.5GB                      |                       8.2GB                        |     ✅      |   
| Qwen-14B  |   23.09.25   |     8K     |             ❎             |          3.0T          |                     18.7GB                      |                       13.0GB                       |     ✅      |
| Qwen-72B  |   23.11.30   |    32K     |             ✅             |          3.0T          |                     61.4GB                      |                       48.9GB                       |     ✅      |   

In this repo, you can figure out:

* Quickstart with Qwen, and enjoy the simple inference.
* Details about the quantization models, including GPTQ and KV cache quantization.
* Statistics of inference performance, including speed and memory.
* Tutorials on finetuning, including full-parameter tuning, LoRA, and Q-LoRA.
* Instructions on deployment, with the example of vLLM and FastChat.
* Instructions on building demos, including WebUI, CLI demo, etc.
* Introduction to DashScope API service, as well as the instructions on building an OpenAI-style API for your model.
* Information about Qwen for tool use, agent, and code interpreter
* Statistics of long-context understanding evaluation
* License agreement
* ...

Also, if you meet problems, turn to [FAQ](FAQ.md) for help first. Still feeling struggled? Feel free to shoot us issues (better in English so that more people can understand you)! If you would like to help us, send us pull requests with no hesitation! We are always excited about PR! 

Would like to chat with us or date us coffee time? Welcome to our Discord or WeChat! 
<br><br>

## News and Updates
* 2023.11.30 🔥 We release **Qwen-72B** and **Qwen-72B-Chat**, which are trained on 3T tokens and support 32k context, along with **Qwen-1.8B**, and **Qwen-1.8B-Chat**, on ModelScope and Hugging Face. We have also strengthened the System Prompt capabilities of the Qwen-72B-Chat and Qwen-1.8B-Chat, see [example documentation](examples/system_prompt.md). Additionally, support the inference on **Ascend 910** and **Hygon DCU**. Check `ascend-support` and `dcu-support` for more details.
* 2023.10.17 We release the Int8 quantized model **Qwen-7B-Chat-Int8** and **Qwen-14B-Chat-Int8**. 
* 2023.9.25 🔥 We release **Qwen-14B** and **Qwen-14B-Chat** on ModelScope and Hugging Face, along with [qwen.cpp](https://github.com/QwenLM/qwen.cpp) and [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent). Codes and checkpoints of **Qwen-7B** and **Qwen-7B-Chat** are also updated. **PLEASE PULL THE LATEST VERSION!**
    - Compared to **Qwen-7B** (original), **Qwen-7B** uses more training tokens, increasing from 2.2T tokens to 2.4T tokens, while the context length extends from 2048 to 8192. The Chinese knowledge and coding ability of **Qwen-7B** have been further improved.
* 2023.9.12 We now support finetuning on the Qwen-7B models, including full-parameter finetuning, LoRA and Q-LoRA.
* 2023.8.21 We release the Int4 quantized model for Qwen-7B-Chat, **Qwen-7B-Chat-Int4**, which requires low memory costs but achieves improved inference speed. Besides, there is no significant performance degradation on the benchmark evaluation.
* 2023.8.3 We release both **Qwen-7B** and **Qwen-7B-Chat** on ModelScope and Hugging Face. We also provide a technical memo for more details about the model, including training details and model performance.
<br>

## Performance
Qwen models outperform the baseline models of similar model sizes on a series of benchmark datasets, e.g., MMLU, C-Eval, GSM8K, MATH, HumanEval, MBPP, BBH, etc., which evaluate the models’ capabilities on natural language understanding, mathematic problem solving, coding, etc. Qwen-72B achieves better performance than LLaMA2-70B on all tasks and outperforms GPT-3.5 on 7 out of 10 tasks. 

<p align="left">
    <img src="assets/radar_72b.jpg" width=600px/>
<p>
      

<br>

| Model             |   MMLU   |  C-Eval  |  GSM8K   |   MATH   | HumanEval |   MBPP   |   BBH    |  CMMLU   |
|:------------------|:--------:|:--------:|:--------:|:--------:|:---------:|:--------:|:--------:|:--------:|
|                   |  5-shot  |  5-shot  |  8-shot  |  4-shot  |  0-shot   |  3-shot  |  3-shot  |  5-shot  |
| LLaMA2-7B         |   46.8   |   32.5   |   16.7   |   3.3    |   12.8    |   20.8   |   38.2   |   31.8   |
| LLaMA2-13B        |   55.0   |   41.4   |   29.6   |   5.0    |   18.9    |   30.3   |   45.6   |   38.4   |
| LLaMA2-34B        |   62.6   |    -     |   42.2   |   6.2    |   22.6    |   33.0   |   44.1   |    -     |
| ChatGLM2-6B       |   47.9   |   51.7   |   32.4   |   6.5    |     -     |    -     |   33.7   |    -     |
| InternLM-7B       |   51.0   |   53.4   |   31.2   |   6.3    |   10.4    |   14.0   |   37.0   |   51.8   |
| InternLM-20B      |   62.1   |   58.8   |   52.6   |   7.9    |   25.6    |   35.6   |   52.5   |   59.0   |
| Baichuan2-7B      |   54.7   |   56.3   |   24.6   |   5.6    |   18.3    |   24.2   |   41.6   |   57.1   |
| Baichuan2-13B     |   59.5   |   59.0   |   52.8   |   10.1   |   17.1    |   30.2   |   49.0   |   62.0   |
| Yi-34B      	  	  |   76.3   |   81.8   |   67.9   |   15.9   |   26.2    |   38.2   |   66.4   |   82.6   |
| XVERSE-65B      	 |   70.8   |   68.6   |   60.3   |    -     |   26.3    |    -     |    -     |    -     |
| **Qwen-1.8B**     |   45.3   |   56.1   |   32.3   |   2.3    |   15.2    |   14.2   |   22.3   |   52.1   |
| **Qwen-7B**       |   58.2   |   63.5   |   51.7   |   11.6   |   29.9    |   31.6   |   45.0   |   62.2   |
| **Qwen-14B**      |   66.3   |   72.1   |   61.3   |   24.8   |   32.3    |   40.8   |   53.4   |   71.0   |
| **Qwen-72B**      | **77.4** | **83.3** | **78.9** | **35.2** | **35.4**  | **52.2** | **67.7** | **83.6** |

For all compared models, we report the best scores between their official reported results and [OpenCompass](https://opencompass.org.cn/leaderboard-llm). 

For more experimental results (detailed model performance on more benchmark datasets) and details, please refer to our technical report by clicking [here](https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf).
<br><br>

## Requirements

* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* transformers 4.32 and above
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
<br>

## Quickstart

Below, we provide simple examples to show how to use Qwen-Chat with 🤖 ModelScope and 🤗 Transformers.

You can use our pre-built docker images to skip most of the environment setup steps, see Section ["Using Pre-built Docker Images"](#-docker) for more details. 

If not using docker, please make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.

```bash
pip install -r requirements.txt
```

If your device supports fp16 or bf16, we recommend installing [flash-attention](https://github.com/Dao-AILab/flash-attention) (**we support flash attention 2 now.**) for higher efficiency and lower memory usage. (**flash-attention is optional and the project can run normally without installing it**)

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# If the version of flash-attn is higher than 2.1.1, the following is not needed.
# pip install csrc/rotary
```

Now you can start with ModelScope or Transformers.

### 🤗 Transformers

To use Qwen-Chat for the inference, all you need to do is to input a few lines of codes as demonstrated below. Remember to pass in the correct model names or paths, such as "Qwen/Qwen-7B-Chat" and "Qwen/Qwen-14B-Chat". However, **please make sure that you are using the latest code.**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Model names: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    device_map="auto",
    trust_remote_code=True
).eval()

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# 3rd dialogue turn
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》
```

Running Qwen, the base language model, is also simple.

<details>
  <summary>Running Qwen</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Model names: "Qwen/Qwen-7B", "Qwen/Qwen-14B" 
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B",
    device_map="auto",
    trust_remote_code=True
).eval()

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# 蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是亚的斯亚贝巴（Addis Ababa）...
```

</details>

<p id="DownloadModel">
In the event of a network issue while attempting to download model checkpoints and codes from HuggingFace, an alternative approach is to initially fetch the checkpoint from ModelScope and then load it from the local directory as outlined below:
</p>

```python
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Downloading model checkpoint to a local dir model_dir
# model_dir = snapshot_download('qwen/Qwen-7B')
# model_dir = snapshot_download('qwen/Qwen-7B-Chat')
# model_dir = snapshot_download('qwen/Qwen-14B')
model_dir = snapshot_download('qwen/Qwen-14B-Chat')

# Loading local checkpoints
# trust_remote_code is still set as True since we still load codes from local dir instead of transformers
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True
).eval()
```

### 🤖 ModelScope

ModelScope is an open-source platform for Model-as-a-Service (MaaS), which provides flexible and cost-effective model service to AI developers. Similarly, you can run the models with ModelScope as shown below:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

# Model names: "qwen/Qwen-7B-Chat", "qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

response, history = model.chat(tokenizer, "你好", history=None)
print(response)
response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=history) 
print(response)
response, history = model.chat(tokenizer, "它有什么好玩的景点", history=history)
print(response)
```

### Batch Inference
Qwen supports batch inference. With flash attention enabled, using batch inference can bring a 40% speedup. The example code is shown below:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

# To generate attention masks automatically, it is necessary to assign distinct
# token_ids to pad_token and eos_token, and set pad_token_id in the generation_config.
tokenizer = AutoTokenizer.from_pretrained(
    './',
    pad_token='<|extra_0|>',
    eos_token='<|endoftext|>',
    padding_side='left',
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    './',
    pad_token_id=tokenizer.pad_token_id,
    device_map="auto",
    trust_remote_code=True
).eval()
model.generation_config = GenerationConfig.from_pretrained('./', pad_token_id=tokenizer.pad_token_id)

all_raw_text = ["我想听你说爱我。", "今天我想吃点啥，甜甜的，推荐下", "我马上迟到了，怎么做才能不迟到"]
batch_raw_text = []
for q in all_raw_text:
    raw_text, _ = make_context(
        tokenizer,
        q,
        system="You are a helpful assistant.",
        max_window_size=model.generation_config.max_window_size,
        chat_format=model.generation_config.chat_format,
    )
    batch_raw_text.append(raw_text)

batch_input_ids = tokenizer(batch_raw_text, padding='longest')
batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(model.device)
batch_out_ids = model.generate(
    batch_input_ids,
    return_dict_in_generate=False,
    generation_config=model.generation_config
)
padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]

batch_response = [
    decode_tokens(
        batch_out_ids[i][padding_lens[i]:],
        tokenizer,
        raw_text_len=len(batch_raw_text[i]),
        context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
        chat_format="chatml",
        verbose=False,
        errors='replace'
    ) for i in range(len(all_raw_text))
]
print(batch_response)

response, _ = model.chat(tokenizer, "我想听你说爱我。", history=None)
print(response)

response, _ = model.chat(tokenizer, "今天我想吃点啥，甜甜的，推荐下", history=None)
print(response)

response, _ = model.chat(tokenizer, "我马上迟到了，怎么做才能不迟到", history=None)
print(response)
```

### CPU

To deploy our models on CPU, we strongly advise you to use [qwen.cpp](https://github.com/QwenLM/qwen.cpp), which is a pure C++ implementation of Qwen and tiktoken. Check the repo for more details!

Also, it is also simple to directly run the model on CPU, which requires your specification of device:

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
```

However, it is likely that you suffer from extremely low inference efficiency.

### Multiple GPUs

If you suffer from lack of GPU memory and you would like to run the model on more than 1 GPU, you can directly use the default loading method, which is now supported by Transformers. The previous method based on `utils.py` is deprecated.

However, though this method is simple, the efficiency of the native pipeline parallelism is low. We advise you to use vLLM with FastChat and please read the section for deployment.

### DashScope
The most simple way to use Qwen through APIs is DashScope API service through Alibaba Cloud. We give an introduction to the usage. Additionally, we provide a script for you to deploy an OpenAI-style API on your own servers.

DashScope is the large language model API service provided by Alibaba Cloud, which now supports Qwen. Note that the models behind DashScope are in-house versions temporarily without details provided. The services include `qwen-turbo` and `qwen-plus`, where the former one runs faster and the latter achieves better performance. For more information, visit the documentation [here](https://dashscope.aliyun.com).

Please head to the official website [link](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.6c2774fahtfXdn) to create a DashScope account and obtain the API key (AK). We recommend setting the AK with an environment variable:
```bash
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
```
Then please install the packages and click [here](https://help.aliyun.com/zh/dashscope/developer-reference/install-dashscope-sdk) for the documentation. If you use Python, you can install DashScope with pip:
```bash
pip install dashscope
```
If you use JAVA SDK, you can install it in this way:
```xml
<!-- https://mvnrepository.com/artifact/com.alibaba/dashscope-sdk-java -->
<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>dashscope-sdk-java</artifactId>
    <version>the-latest-version</version>
</dependency>
```
The simplest way to use DashScope is the usage with messages, which is similar to OpenAI API. The example is demonstrated below:
```python
import random
from http import HTTPStatus
from dashscope import Generation


def call_with_messages():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '如何做西红柿鸡蛋？'}]
    gen = Generation()
    response = gen.call(
        Generation.Models.qwen_turbo,
        messages=messages,
        seed=random.randint(1, 10000),  # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message" format.
    )
    return response


if __name__ == '__main__':
    response = call_with_messages()
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
```
For more usages, please visit the official website for more details.
<br><br>

## Quantization

### GPTQ

We provide a solution based on [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), and release the Int4 and Int8 quantized models, which achieve nearly lossless model effects but improved performance on both memory costs and inference speed.

Here we demonstrate how to use our provided quantized models for inference. Before you start, make sure you meet the requirements of auto-gptq (e.g., torch 2.0 and above, transformers 4.32.0 and above, etc.) and install the required packages:

```bash
pip install auto-gptq optimum
```

If you meet problems installing `auto-gptq`, we advise you to check out the official [repo](https://github.com/PanQiWei/AutoGPTQ) to find a wheel.

> Note: The pre-compiled `auto-gptq` packages strongly depend on the version of `torch` and its CUDA version. Moreover, due to recent update, 
> you may also encounter unsupported version errors from `transformers`, `optimum`, or `peft`.
> We recommend using the latest versions meeting the following requirements:
> - torch==2.1 auto-gptq>=0.5.1 transformers>=4.35.0 optimum>=1.14.0 peft>=0.6.1
> - torch>=2.0,<2.1 auto-gptq<0.5.0 transformers<4.35.0 optimum<1.14.0 peft>=0.5.0,<0.6.0

Then you can load the quantized model easily and run inference as same as usual:

```python
# Model names: "Qwen/Qwen-7B-Chat-Int4", "Qwen/Qwen-14B-Chat-Int4"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "Hi", history=None)
```

We illustrate the model performance of both BF16, Int8 and Int4 models on the benchmark, and we find that the quantized model does not suffer from significant performance degradation. Results are shown below:

| Quantization         | MMLU | CEval (val) | GSM8K | Humaneval |
|----------------------|:----:|:-----------:|:-----:|:---------:|
| Qwen-1.8B-Chat (BF16)| 43.3 |    55.6     | 33.7  |   26.2    |
| Qwen-1.8B-Chat (Int8)| 43.1 |    55.8     | 33.0  |   27.4    |
| Qwen-1.8B-Chat (Int4)| 42.9 |    52.8     | 31.2  |   25.0    |
| Qwen-7B-Chat (BF16)  | 55.8 |    59.7     | 50.3  |   37.2    |
| Qwen-7B-Chat (Int8)  | 55.4 |    59.4     | 48.3  |   34.8    |
| Qwen-7B-Chat (Int4)  | 55.1 |    59.2     | 49.7  |   29.9    |
| Qwen-14B-Chat (BF16) | 64.6 |    69.8     | 60.1  |   43.9    |
| Qwen-14B-Chat (Int8) | 63.6 |    68.6     | 60.0  |   48.2    |
| Qwen-14B-Chat (Int4) | 63.3 |    69.0     | 59.8  |   45.7    |
| Qwen-72B-Chat (BF16) | 74.4 |    80.1     | 76.4  |   64.6    |
| Qwen-72B-Chat (Int8) | 73.5 |    80.1     | 73.5  |   62.2    |
| Qwen-72B-Chat (Int4) | 73.4 |    80.1     | 75.3  |   61.6    |

### Quantization of KV cache

> NOTE: Please be aware that due to the internal mechanism of Hugging Face, the support files for this functionality 
> (i.e., `cache_autogptq_cuda_256.cpp` and `cache_autogptq_cuda_kernel_256.cu`) may be missing. Please manually download
> them from the Hugging Face Hub and place them into the same folder as the other module files.

The attention KV cache can be quantized and compressed for storage, to get a higher sample throughput. The arguments `use_cache_quantization` and `use_cache_kernel` in `config.json` are provided to enable KV cache quantization. The specific use method is as follows:
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
     device_map="auto",
     trust_remote_code=True,
     use_cache_quantization=True,
     use_cache_kernel=True,
     use_flash_attn=False
)
```
Attention: Currently, KV cache quantization and flash attention cannot be used at the same time.
If you enable KV cache quantization and flash attention at the same time (`use_flash_attn=True`, `use_cache_quantization=True`, `use_cache_kernel=True`), `use_flash_attn` is disabled by default (`use_flash_attn=false`).

We have verified that the use of the quantized Int8-KV-Cache model does not suffer from significant performance degradation in downstream evaluation. In the following, we focus on profiling its memory footprint in different conditions. 
The profiling runs on a single A100-SXM4-80G GPU with PyTorch 2.0.1 and CUDA 11.4. 
We use BF16 models to generate 1024 tokens by default, and "OOM" indicates out-of-memory error.

With KV cache quantization, the model can infer with a larger batch size (bs).

| USE KV Cache |  bs=1  |  bs=4  | bs=16  | bs=32  | bs=64  | bs=100 |
|--------------|:------:|:------:|:------:|:------:|:------:|:------:|
| No           | 16.3GB | 24.1GB | 31.7GB | 48.7GB |  OOM   |  OOM   |
| Yes          | 15.5GB | 17.2GB | 22.3GB | 30.2GB | 48.2GB | 72.4GB |

With KV cache quantization the model can save more memory when generating longer sequence (`sl`, sequence length, referring to the number of tokens generated) at the stage of inference.

| USE KV Cache | sl=512 | sl=1024 | sl=2048 | sl=4096 | sl=8192 |
|--------------|:------:|:-------:|:-------:|:-------:|:-------:|
| No           | 15.2GB | 16.3GB  | 17.6GB  | 19.5GB  | 23.2GB  |
| Yes          |  15GB  | 15.5GB  | 15.8GB  | 16.6GB  | 17.6GB  |

The model with KV cache quantization will convert the format of `layer_past` from float to int8, and meanwhile the quantized `layer-past` will also store the quantization parameters.

Specific steps are as follows:

1. Quantize key/value
```
    qv,scale,zero_point=quantize_cache_v(v)
```
2. Store into layer_past

The following is the format of quantized `layer_past`:
```
    layer_past=((q_key,key_scale,key_zero_point),
                (q_value,value_scale,value_zero_point))
```

The original format of `layer_past` is shown below:
```
    layer_past=(key,value)
```

If you want to use the attention KV which is quantized, you can use the dequantization operation to convert the Int8 key/value back to the float format as follows:
```
    v=dequantize_cache_torch(qv,scale,zero_point)
```
<br>


## Inference Performance

This section provides the statistics of speed and memory of models in different precisions. The speed and memory profiling are conducted using [this script](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py). 

We measured the average inference speed (tokens/s) and GPU memory usage of generating 2048 with the models in BF16, Int8, and Int4. 

<table>
    <tr>
        <td>Model Size</td>
        <td>Quantization</td>
        <td>Speed (Tokens/s)</td>
        <td>GPU Memory Usage</td>
    </tr>
    <tr>
        <td rowspan="3">1.8B</td>
        <td>BF16</td>
        <td>54.09</td>
        <td>4.23GB</td>
    </tr>
    <tr>
        <td>Int8</td>
        <td>55.56</td>
        <td>3.48GB</td>
    </tr>
    <tr>
        <td>Int4</td>
        <td>71.07</td>
        <td>2.91GB</td>
    </tr>
    <tr>
        <td rowspan="3">7B</td>
        <td>BF16</td>
        <td>40.93</td>
        <td>16.99GB</td>
    </tr>
    <tr>
        <td>Int8</td>
        <td>37.47</td>
        <td>11.20GB</td>
    </tr>
    <tr>
        <td>Int4</td>
        <td>50.09</td>
        <td>8.21GB</td>
    </tr>
    <tr>
        <td rowspan="3">14B</td>
        <td>BF16</td>
        <td>32.22</td>
        <td>30.15GB</td>
    </tr>
    <tr>
        <td>Int8</td>
        <td>29.28</td>
        <td>18.81GB</td>
    </tr>
    <tr>
        <td>Int4</td>
        <td>38.72</td>
        <td>13.01GB</td>
    </tr>
    <tr>
        <td rowspan="3">72B</td>
        <td>BF16</td>
        <td>8.48</td>
        <td>144.69GB (2xA100)</td>
    </tr>
    <tr>
        <td>Int8</td>
        <td>9.05</td>
        <td>81.27GB (2xA100)</td>
    </tr>
    <tr>
        <td>Int4</td>
        <td>11.32</td>
        <td>48.86GB</td>
    </tr>
    <tr>
        <td>72B + vLLM</td>
        <td>BF16</td>
        <td>17.60</td>
        <td>2xA100</td>
    </tr>
</table>

The profiling runs on a single A100-SXM4-80G GPU (except 2xA100 is mentioned) with PyTorch 2.0.1, CUDA 11.8, and Flash-Attention 2. (72B + vLLM uses PyTorch 2.1.0 and Cuda 11.8.) The inference speed is averaged over the encoded and generated tokens.

Note: The generation speed of the Int4/Int8 models mentioned above is provided by the autogptq library. The current speed of the model loaded using ``AutoModelForCausalLM.from_pretrained`` will be approximately 20% slower. We have reported this issue to the HuggingFace team and will update it promptly if a solution is available.

We also measure the inference speed and GPU memory usage with different settings of context and generation lengths, Flash-Attention version. You can find the results in the according modelcards on Hugging Face or ModelScope.

## Finetuning

### Usage
Now we provide the official training script, `finetune.py`, for users to finetune the pretrained model for downstream applications in a simple fashion. Additionally, we provide shell scripts to launch finetuning with no worries. This script supports the training with [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/). The shell scripts that we provide use DeepSpeed (Note: this may have conflicts with the latest version of pydantic and you should use make sure `pydantic<2.0`) and Peft. You can install them by:
```bash
pip install peft deepspeed
```

To prepare your training data, you need to put all the samples into a list and save it to a json file. Each sample is a dictionary consisting of an id and a list for conversation. Below is a simple example list with 1 sample:
```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是一个语言模型，我叫通义千问。"
      }
    ]
  }
]
```

After data preparation, you can use the provided shell scripts to run finetuning. Remember to specify the path to the data file, `$DATA`.

The finetuning scripts allow you to perform:
- Full-parameter finetuning
- LoRA
- Q-LoRA

Full-parameter finetuning requires updating all parameters in the whole training process. To launch your training, run the following script:

```bash
# Distributed training. We do not provide single-GPU training script as the insufficient GPU memory will break down the training.
bash finetune/finetune_ds.sh
```

Remember to specify the correct model name or path, the data path, as well as the output directory in the shell scripts. Another thing to notice is that we use DeepSpeed ZeRO 3 in this script. If you want to make changes, just remove the argument `--deepspeed` or make changes in the DeepSpeed configuration json file based on your requirements. Additionally, this script supports mixed-precision training, and thus you can use `--bf16 True` or `--fp16 True`. Remember to use DeepSpeed when you use fp16 due to mixed precision training. Empirically we advise you to use bf16 to make your training consistent with our pretraining and alignment if your machine supports bf16, and thus we use it by default.

Similarly, to run LoRA, use another script to run as shown below. Before you start, make sure that you have installed `peft`. Also, you need to specify your paths to your model, data, and output. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load. Also, this script support both bf16 and fp16.

```bash
# Single GPU training
bash finetune/finetune_lora_single_gpu.sh
# Distributed training
bash finetune/finetune_lora_ds.sh
```

In comparison with full-parameter finetuning, LoRA ([paper](https://arxiv.org/abs/2106.09685)) only updates the parameters of adapter layers but keeps the original large language model layers frozen. This allows much fewer memory costs and thus fewer computation costs. 

Note that if you use LoRA to finetune the base language model, e.g., Qwen-7B, instead of chat models, e.g., Qwen-7B-Chat, the script automatically switches the embedding and output layer as trainable parameters. This is because the base language model has no knowledge of special tokens brought by ChatML format. Thus these layers should be updated for the model to understand and predict the tokens. Or in another word, if your training brings in special tokens in LoRA, you should set the layers to trainable parameters by setting `modules_to_save` inside the code. Also, if we have these parameters trainable, it is not available to use ZeRO 3, and this is why we use ZeRO 2 in the script by default. If you do not have new trainable parameters, you can switch to ZeRO 3 by changing the DeepSpeed configuration file. Additionally, we find that there is a significant gap between the memory footprint of LoRA with and without these trainable parameters. Therefore, if you have trouble with memory, we advise you to LoRA finetune the chat models. Check the profile below for more information. 

If you still suffer from insufficient memory, you can consider Q-LoRA ([paper](https://arxiv.org/abs/2305.14314)), which uses the quantized large language model and other techniques such as paged attention to allow even fewer memory costs. 

Note: to run single-GPU Q-LoRA training, you may need to install `mpi4py` through `pip` or `conda`.

To run Q-LoRA, directly run the following script:

```bash
# Single GPU training
bash finetune/finetune_qlora_single_gpu.sh
# Distributed training
bash finetune/finetune_qlora_ds.sh
```

For Q-LoRA, we advise you to load our provided quantized model, e.g., Qwen-7B-Chat-Int4. You **SHOULD NOT** use the bf16 models. Different from full-parameter finetuning and LoRA, only fp16 is supported for Q-LoRA. For single-GPU training, we have to use DeepSpeed for mixed-precision training due to our observation of errors caused by torch amp. Besides, for Q-LoRA, the troubles with the special tokens in LoRA still exist. However, as we only provide the Int4 models for chat models, which means the language model has learned the special tokens of ChatML format, you have no worry about the layers. Note that the layers of the Int4 model should not be trainable, and thus if you introduce special tokens in your training, Q-LoRA might not work.

> NOTE: Please be aware that due to the internal mechanisms of Hugging Face, certain non-Python files (e.g., `*.cpp` and `*.cu`) 
> may be missing from the saved checkpoint. You may need to manually copy them to the directory containing other files.

Different from full-parameter finetuning, the training of both LoRA and Q-LoRA only saves the adapter parameters. Suppose your training starts from Qwen-7B, you can load the finetuned model for inference as shown below:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()
```

> NOTE: If `peft>=0.8.0`, it will try to load the tokenizer as well, however, initialized without `trust_remote_code=True`, leading to `ValueError: Tokenizer class QWenTokenizer does not exist or is not currently imported.` Currently, you could downgrade `peft<0.8.0` or move tokenizer files elsewhere to workaround this issue.

If you want to merge the adapters and save the finetuned model as a standalone model (you can only do this with LoRA, and you CANNOT merge the parameters from Q-LoRA), you can run the following codes:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
```

The `new_model_directory` directory will contain the merged model weights and module files. Please note that `*.cu` and `*.cpp` files may be missing in the saved files. If you wish to use the KV cache functionality, please manually copy them. Besides, the tokenizer files are not saved in the new directory in this step. You can copy the tokenizer files or use the following code
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)

tokenizer.save_pretrained(new_model_directory)
```


Note: For multi-GPU training, you need to specify the proper hyperparameters for distributed training based on your machine. Besides, we advise you to specify your maximum sequence length with the argument `--model_max_length`, based on your consideration of data, memory footprint, and training speed.

### Quantize Fine-tuned Models

This section applies to full-parameter/LoRA fine-tuned models. (Note: You do not need to quantize the Q-LoRA fine-tuned model because it is already quantized.)
If you use LoRA, please follow the above instructions to merge your model before quantization. 

We recommend using [auto_gptq](https://github.com/PanQiWei/AutoGPTQ) to quantize the finetuned model. 

```bash
pip install auto-gptq optimum
```

Note: Currently AutoGPTQ has a bug referred in [this issue](https://github.com/PanQiWei/AutoGPTQ/issues/370). Here is a [workaround PR](https://github.com/PanQiWei/AutoGPTQ/pull/495), and you can pull this branch and install from the source.

First, prepare the calibration data. You can reuse the fine-tuning data, or use other data following the same format.

Second, run the following script:

```bash
python run_gptq.py \
    --model_name_or_path $YOUR_LORA_MODEL_PATH \
    --data_path $DATA \
    --out_path $OUTPUT_PATH \
    --bits 4 # 4 for int4; 8 for int8
```

This step requires GPUs and may costs a few hours according to your data size and model size.

Then, copy all `*.py`, `*.cu`, `*.cpp` files and `generation_config.json` to the output path. And we recommend you to overwrite `config.json` by copying the file from the coresponding official quantized model
(for example, if you are fine-tuning `Qwen-7B-Chat` and use `--bits 4`, you can find the `config.json` from [Qwen-7B-Chat-Int4](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blob/main/config.json)).
You should also rename the ``gptq.safetensors`` into ``model.safetensors``.

Finally, test the model by the same method to load the official quantized model. For example,

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/your/model",
    device_map="auto",
    trust_remote_code=True
).eval()

response, history = model.chat(tokenizer, "你好", history=None)
print(response)
```

### Multinode Finetuning

Our provided scripts support multinode finetuning. You can refer to the comments in [script](./finetune/finetune_lora_ds.sh) to correctly set corresponding arguments and launch the script on each node. For more information about multinode distributed training, please refer to [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

Note: DeepSpeed ZeRO 3 requires much greater inter-node communication rate than ZeRO 2, which will significantly reduce the training speed in the case of multinode finetuning. Therefore, we do not recommend using DeepSpeed ZeRO 3 configurations in multinode finetuning scripts.

### Profiling of Memory and Speed
We profile the GPU memory and training speed of both LoRA (LoRA (emb) refers to training the embedding and output layer, while LoRA has no trainable embedding and output layer) and Q-LoRA in the setup of single-GPU training. In this test, we experiment on a single A100-SXM4-80G GPU, and we use CUDA 11.8 and Pytorch 2.0. Flash attention 2 is applied. We uniformly use a batch size of 1 and gradient accumulation of 8. We profile the memory (GB) and speed (s/iter) of inputs of different lengths, namely 256, 512, 1024, 2048, 4096, and 8192. We also report the statistics of full-parameter finetuning with Qwen-7B on 2 A100 GPUs. We only report the statistics of 256, 512, and 1024 tokens due to the limitation of GPU memory. 

For Qwen-7B, we also test the performance of multinode finetuning. We experiment using two servers, each containing two A100-SXM4-80G GPUs, and the rest of configurations are the same as other Qwen-7B experiments. The results of multinode finetuning are marked as LoRA (multinode) in the table.

For Qwen-72B, we experiment in two ways: 1) Lora fintuning + DeepSpeed ZeRO 3 on 4 A100-SXM4-80G GPUs and 2) QLora (int4) fine-tuning on a single A100-SXM4-80G GPU. Note that OOM occurs on 4 A100-SXM4-80G GPUs both with LoRA (emb) fine-tuning and LoRA fine-tuning without Deepspeed ZeRO 3 (you can pass `--deepspeed finetune/ds_config_zero3.json` to [`finetune/finetune_lora_ds.sh`](finetune/finetune_lora_ds.sh) to enable DeepSpeed ZeRO 3).

The statistics are listed below:

<table>
    <tr>
      <th rowspan="2">Model Size</th><th rowspan="2">Method</th><th rowspan="2">#Nodes</th><th rowspan="2">#GPUs per node</th><th colspan="6" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">256</th><th align="center">512</th><th align="center">1024</th><th align="center">2048</th><th align="center">4096</th><th align="center">8192</th>
    </tr>
    <tr>
        <th rowspan="4">1.8B</th><td>LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">6.7G / 1.0s/it</td><td align="center">7.4G / 1.0s/it</td><td align="center">8.4G / 1.1s/it</td><td align="center">11.0G / 1.7s/it</td><td align="center">16.2G / 3.3s/it</td><td align="center">21.8G / 6.8s/it</td>
    </tr>
    <tr>
        <td>LoRA (emb)</td>
        <td>1</td><td>1</td>
        <td align="center">13.7G / 1.0s/it</td><td align="center">14.0G / 1.0s/it</td><td align="center">14.0G / 1.1s/it</td><td align="center">15.1G / 1.8s/it</td><td align="center">19.7G / 3.4s/it</td><td align="center">27.7G / 7.0s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">5.8G / 1.4s/it</td><td align="center">6.0G / 1.4s/it</td><td align="center">6.6G / 1.4s/it</td><td align="center">7.8G / 2.0s/it</td><td align="center">10.2G / 3.4s/it</td><td align="center">15.8G / 6.5s/it</td>
    </tr>
    <tr>
        <td>Full-parameter</td>
        <td>1</td><td>1</td>
        <td align="center">43.5G / 2.1s/it</td><td align="center">43.5G / 2.2s/it</td><td align="center">43.5G / 2.2s/it</td><td align="center">43.5G / 2.3s/it</td><td align="center">47.1G / 2.8s/it</td><td align="center">48.3G / 5.6s/it</td>
    </tr>
    <tr>
        <th rowspan="5">7B</th>
        <td>LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">20.1G / 1.2s/it</td><td align="center">20.4G / 1.5s/it</td><td align="center">21.5G / 2.8s/it</td><td align="center">23.8G / 5.2s/it</td><td align="center">29.7G / 10.1s/it</td><td align="center">36.6G / 21.3s/it</td>
    </tr>
    <tr>
        <td>LoRA (emb)</td>
        <td>1</td><td>1</td>
        <td align="center">33.7G / 1.4s/it</td><td align="center">34.1G / 1.6s/it</td><td align="center">35.2G / 2.9s/it</td><td align="center">35.1G / 5.3s/it</td><td align="center">39.2G / 10.3s/it</td><td align="center">48.5G / 21.7s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">11.5G / 3.0s/it</td><td align="center">11.5G / 3.0s/it</td><td align="center">12.3G / 3.5s/it</td><td align="center">13.9G / 7.0s/it</td><td align="center">16.9G / 11.6s/it</td><td align="center">23.5G / 22.3s/it</td>
    </tr>
    <tr>
        <td>Full-parameter</td>
<td>1</td><td>2</td>
<td align="center">139.2G / 4.0s/it</td><td align="center">148.0G / 4.0s/it</td><td align="center">162.0G / 4.5s/it</td><td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
        <td>LoRA (multinode)</td>
        <td>2</td><td>2</td>
        <td align="center">74.7G / 2.09s/it</td><td align="center">77.6G / 3.16s/it</td><td align="center">84.9G / 5.17s/it</td><td align="center">95.1G / 9.25s/it</td><td align="center">121.1G / 18.1s/it</td><td align="center">155.5G / 37.4s/it</td>
    </tr>
    <tr>
        <th rowspan="3">14B</th>
        <td>LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">34.6G / 1.6s/it</td><td align="center">35.1G / 2.4s/it</td><td align="center">35.3G / 4.4s/it</td><td align="center">37.4G / 8.4s/it</td><td align="center">42.5G / 17.0s/it</td><td align="center">55.2G / 36.0s/it</td>
    </tr>
    <tr>
        <td>LoRA (emb)</td>
        <td>1</td><td>1</td>
        <td align="center">51.2 / 1.7s/it</td><td align="center">51.1G / 2.6s/it</td><td align="center">51.5G / 4.6s/it</td><td align="center">54.1G / 8.6s/it</td><td align="center">56.8G / 17.2s/it</td><td align="center">67.7G / 36.3s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">18.7G / 5.3s/it</td><td align="center">18.4G / 6.3s/it</td><td align="center">18.9G / 8.2s/it</td><td align="center">19.9G / 11.8s/it</td><td align="center">23.0G / 20.1s/it</td><td align="center">27.9G / 38.3s/it</td>
    </tr>
    <tr>
        <th rowspan="2">72B</th>
        <td>LoRA + Deepspeed Zero3</td>
        <td>1</td><td>4</td>
        <td align="center">215.4G / 17.6s/it</td><td align="center">217.7G / 20.5s/it</td><td align="center">222.6G / 29.4s/it</td><td align="center">228.8G / 45.7s/it</td><td align="center">249.0G / 83.4s/it</td><td align="center">289.2G / 161.5s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">61.4G / 27.4s/it</td><td align="center">61.4G / 31.5s/it</td><td align="center">62.9G / 41.4s/it</td><td align="center">64.1G / 59.5s/it</td><td align="center">68.0G / 97.7s/it</td><td align="center">75.6G / 179.8s/it</td>
    </tr>
</table>

<br>

## Deployment

### vLLM 

For deployment and fast inference, we suggest using vLLM. 

If you use cuda 12.1 and pytorch 2.1, you can directly use the following command to install vLLM.

```bash
# pip install vllm  # This line is faster but it does not support quantization models.

# The below lines support int4 quantization (int8 will be supported soon). The installation are slower (~10 minutes).
git clone https://github.com/QwenLM/vllm-gptq
cd vllm-gptq
pip install -e .
```

Otherwise, please refer to the official vLLM [Installation Instructions](https://docs.vllm.ai/en/latest/getting_started/installation.html), or our [vLLM repo for GPTQ quantization](https://github.com/QwenLM/vllm-gptq).

#### vLLM + Transformer-like Wrapper

You can download the [wrapper codes](examples/vllm_wrapper.py) and execute the following commands for multiple rounds of dialogue interaction. (Note: It currently only supports the ``model.chat()`` method.)

```python
from vllm_wrapper import vLLMWrapper

model = vLLMWrapper('Qwen/Qwen-7B-Chat', tensor_parallel_size=1)
# model = vLLMWrapper('Qwen/Qwen-7B-Chat-Int4', tensor_parallel_size=1, dtype="float16")

response, history = model.chat(query="你好", history=None)
print(response)
response, history = model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history = model.chat(query="给这个故事起一个标题", history=history)
print(response)
```

#### vLLM + Web Demo / OpenAI-like API

You can use FastChat to lauch a web demo or an OpenAI API server. First, install FastChat:

```bash
pip install "fschat[model_worker,webui]"
```

To run Qwen with vLLM and FastChat, you need launch a controller by:
```bash
python -m fastchat.serve.controller
```

Then you can launch the model worker, which means loading your model for inference. For single GPU inference, you can directly run:
```bash
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --dtype bfloat16
# python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --dtype float16 # run int4 model
```
However, if you hope to run the model on multiple GPUs for faster inference or larger memory, you can use tensor parallelism supported by vLLM. Suppose you run the model on 4 GPUs, the command is shown below:
```bash
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --tensor-parallel-size 4 --dtype bfloat16
# python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --tensor-parallel-size 4 --dtype float16 # run int4 model
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

However, if you find it difficult to use vLLM and FastChat, you can try our provided simplest methods to deploy a web demo, CLI demo, and API.


### Web UI

We provide code for users to build a web UI demo (thanks to @wysaid). Before you start, make sure you install the following packages:

```
pip install -r requirements_web_demo.txt
```

Then run the command below and click on the generated link:

```bash
python web_demo.py
```

<p align="center">
    <br>
    <img src="assets/web_demo.gif" width="600" />
    <br>
<p>

### CLI Demo

We provide a CLI demo example in `cli_demo.py`, which supports streaming output for the generation. Users can interact with Qwen-7B-Chat by inputting prompts, and the model returns model outputs in the streaming mode. Run the command below:

```bash
python cli_demo.py
```

<p align="center">
    <br>
    <img src="assets/cli_demo.gif" width="600" />
    <br>
<p>
<br>

### API

We provide methods to deploy local API based on OpenAI API (thanks to @hanpenggit). Before you start, install the required packages:

```bash
pip install fastapi uvicorn "openai<1.0" pydantic sse_starlette
```

Then run the command to deploy your API:

```bash
python openai_api.py
```

You can change your arguments, e.g., `-c` for checkpoint name or path, `--cpu-only` for CPU deployment, etc. If you meet problems launching your API deployment, updating the packages to the latest version can probably solve them.

Using the API is also simple. See the example below:

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

<p align="center">
    <br>
    <img src="assets/openai_api.gif" width="600" />
    <br>
<p>

**Function calling** is also supported (but only when `stream=False` for the moment). See the [example usage](examples/function_call_examples.py) here.
<br><br>

## 🐳 Docker

To simplify the deployment process, we provide docker images with pre-built environments: [qwenllm/qwen](https://hub.docker.com/r/qwenllm/qwen). You only need to install the driver and download model files to launch demos, deploy OpenAI API, and finetune the model.

### Preparation

1. Install the correct version of Nvidia driver depending on the image to use:
  - `qwenllm/qwen:cu117` (**recommend**): `>= 515.48.07`
  - `qwenllm/qwen:cu114` (w/o flash-attention): `>= 470.82.01`
  - `qwenllm/qwen:cu121`: `>= 530.30.02`
  - `qwenllm/qwen:latest`: same as `qwenllm/qwen:cu117`

2. Install and configure [docker](https://docs.docker.com/engine/install/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
# configure docker
sudo systemctl start docker
# test if docker is correctly installed
sudo docker run hello-world

# configure nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# test if nvidia-container-toolkit is correctly installed
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

3. Download model checkpoints and codes to your environment (see [here](#DownloadModel)).

### Deployment

Here we use Qwen-7B-Chat as an example. Before launching a web demo or API, you can setup the configuration as shown below:

```bash
IMAGE_NAME=qwenllm/qwen:cu117
PORT=8901
CHECKPOINT_PATH=/path/to/Qwen-7B-Chat   # Path to downloaded model checkpoints and codes
```
The following scripts can help you build:

* OpenAI API
```bash
bash docker/docker_openai_api.sh -i ${IMAGE_NAME} -c ${CHECKPOINT_PATH} --port ${PORT}
```

* Web UI
```bash
bash docker/docker_web_demo.sh -i ${IMAGE_NAME} -c ${CHECKPOINT_PATH} --port ${PORT}
```

* CLI Demo
```bash
bash docker/docker_cli_demo.sh -i ${IMAGE_NAME} -c ${CHECKPOINT_PATH}
```

The commands above will automatically download the required image and launch a Web UI demo in background (the service will auto-restart). You can open `http://localhost:${PORT}` on the host to use the demo.

The demo is successfully launched if you see the following output:

```text
Successfully started web demo. Open '...' to try!
Run `docker logs ...` to check demo status.
Run `docker rm -f ...` to stop and remove the demo.
```

If you want to check the status of the demo, you can use `docker logs qwen` to display outputs.

You can use `docker rm -f qwen` to stop the service and remove the container.


### Finetuning

The method of finetuning using the pre-built Docker image is basically the same as [the above chapter](#Finetuning) (we have already installed dependencies in the image):

The following is an example of single-GPU LoRA:
```bash
IMAGE_NAME=qwenllm/qwen:cu117
CHECKPOINT_PATH=/path/to/Qwen-7B                # Path to downloaded model checkpoints and codes
#CHECKPOINT_PATH=/path/to/Qwen-7B-Chat-Int4     # Path to downloaded model checkpoints and codes (Q-LoRA)
DATA_PATH=/path/to/data/root                    # Prepare finetune data at ${DATA_PATH}/example.json
OUTPUT_PATH=/path/to/output/checkpoint          # Path to finetune outputs

# Use all host devices by default
DEVICE=all
# If you need to specify GPUs for training, set device as follow (NOTE: internal quotation marks cannot be omitted)
#DEVICE='"device=0,1,2,3"'

mkdir -p ${OUTPUT_PATH}

# Single-GPU LoRA finetuning
docker run --gpus ${DEVICE} --rm --name qwen \
    --mount type=bind,source=${CHECKPOINT_PATH},target=/data/shared/Qwen/Qwen-7B \
    --mount type=bind,source=${DATA_PATH},target=/data/shared/Qwen/data \
    --mount type=bind,source=${OUTPUT_PATH},target=/data/shared/Qwen/output_qwen \
    --shm-size=2gb \
    -it ${IMAGE_NAME} \
    bash finetune/finetune_lora_single_gpu.sh -m /data/shared/Qwen/Qwen-7B/ -d /data/shared/Qwen/data/example.json
```

To make a change to single-GPU Q-LoRA for example, you just need to modify the bash command inside `docker run`:
```bash
bash finetune/finetune_qlora_single_gpu.sh -m /data/shared/Qwen/Qwen-7B-Chat-Int4/ -d /data/shared/Qwen/data/example.json
```
<br>

## 🔥 System Prompt
Qwen-1.8-Chat and Qwen-72B-Chat have been fully trained on diverse system prompts with multiple rounds of complex interactions, so that they can follow a variety of system prompts and realize model customization in context, further improving the scalability of Qwen-chat.

With System Prompt, Qwen-Chat can realize **roly playing**, **language style transfer**, **task setting**, and **behavior setting**.

![](assets/system_prompt_language_style.png)

![](assets/system_prompt_role_play_en.png)

For more information, please refer to the [example documentation](examples/system_prompt.md).

## Tool Usage

Qwen-Chat has been optimized for tool usage and function calling capabilities. Users can develop agents, LangChain applications, and even augment Qwen with a Python Code Interpreter.

We provide documentation on how to implement tool calls based on the principle of ReAct Prompting, please refer to [the ReAct example](examples/react_prompt.md). Based on this principle, we provide support for function calling in [openai_api.py](openai_api.py).

We have tested the model's tool calling capabilities on our open-source Chinese evaluation benchmark and found that Qwen-Chat consistently performs well:

<table>
    <tr>
        <th colspan="4" align="center">Chinese Tool-Use Benchmark (Version 20231206)</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Tool Selection (Acc.↑)</th><th align="center">Tool Input (Rouge-L↑)</th><th align="center">False Positive Error↓</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">98.0%</td><td align="center">0.953</td><td align="center">23.9%</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">74.5%</td><td align="center">0.807</td><td align="center">80.6%</td>
    </tr>
    <tr>
        <td>Qwen-1_8B-Chat</td><td align="center">85.0%</td><td align="center">0.839</td><td align="center">27.6%</td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td><td align="center">95.5%</td><td align="center">0.900</td><td align="center">11.6%</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">96.9%</td><td align="center">0.917</td><td align="center">5.6%</td>
    </tr>
    <tr>
        <td>Qwen-72B-Chat</td><td align="center">98.2%</td><td align="center">0.927</td><td align="center">1.1%</td>
    </tr>
</table>

To assess Qwen's ability to use the Python Code Interpreter for tasks such as mathematical problem solving, data visualization, and other general-purpose tasks such as file handling and web scraping, we have created and open-sourced a benchmark specifically designed for evaluating these capabilities. You can find the benchmark at this [link](https://github.com/QwenLM/Qwen-Agent/tree/main/benchmark).

We have observed that Qwen performs well in terms of code executability and result accuracy when generating code:

<table>
    <tr>
        <th colspan="5" align="center">Code Interpreter Benchmark (Version 20231206)</th>
    </tr>
    <tr>
        <th rowspan="2" align="center">Model</th>
        <th colspan="3" align="center">Accuracy of Code Execution Results (%)</th>
        <th colspan="1" align="center">Executable Rate of Code (%)</th>
    </tr>
    <tr>
        <th align="center">Math↑</th><th align="center">Visualization-Hard↑</th><th align="center">Visualization-Easy↑</th><th align="center">General↑</th>
    </tr>
    <tr>
        <td>GPT-4</td>
        <td align="center">82.8</td>
        <td align="center">66.7</td>
        <td align="center">60.8</td>
        <td align="center">82.8</td>
    </tr>
    <tr>
        <td>GPT-3.5</td>
        <td align="center">47.3</td>
        <td align="center">33.3</td>
        <td align="center">55.7</td>
        <td align="center">74.1</td>
    </tr>
    <tr>
        <td>LLaMA2-13B-Chat</td>
        <td align="center">8.3</td>
        <td align="center">1.2</td>
        <td align="center">15.2</td>
        <td align="center">48.3</td>
    </tr>
    <tr>
        <td>CodeLLaMA-13B-Instruct</td>
        <td align="center">28.2</td>
        <td align="center">15.5</td>
        <td align="center">21.5</td>
        <td align="center">74.1</td>
    </tr>
    <tr>
        <td>InternLM-20B-Chat</td>
        <td align="center">34.6</td>
        <td align="center">10.7</td>
        <td align="center">25.1</td>
        <td align="center">65.5</td>
    </tr>
    <tr>
        <td>ChatGLM3-6B</td>
        <td align="center">54.2</td>
        <td align="center">4.8</td>
        <td align="center">15.2</td>
        <td align="center">67.1</td>
    </tr>
    <tr>
        <td>Qwen-1.8B-Chat</td>
        <td align="center">25.6</td>
        <td align="center">21.4</td>
        <td align="center">22.8</td>
        <td align="center">65.5</td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td>
        <td align="center">41.9</td>
        <td align="center">23.8</td>
        <td align="center">38.0</td>
        <td align="center">67.2</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td>
        <td align="center">58.4</td>
        <td align="center">31.0</td>
        <td align="center">45.6</td>
        <td align="center">65.5</td>
    </tr>
    <tr>
        <td>Qwen-72B-Chat</td>
        <td align="center">72.7</td>
        <td align="center">41.7</td>
        <td align="center">43.0</td>
        <td align="center">82.8</td>
    </tr>
</table>

<p align="center">
    <br>
    <img src="assets/code_interpreter_showcase_001.jpg" />
    <br>
<p>

<br>

## Long-Context Understanding

To extend the context length and break the bottleneck of training sequence length, we introduce several techniques, including NTK-aware interpolation, window attention, and LogN attention scaling, to extend the context length of Qwen-14B from 2K to over 8K tokens, and Qwen-1.8B/7B from 8K to 32K tokens. 

For Qwen-72B, we adapt RoPE to longer contexts with a larger rotary base. Qwen-72B supports the max context length of 32K tokens.

We conduct language modeling experiments on the arXiv dataset with the PPL evaluation and find that Qwen can reach outstanding performance in the scenario of long context. Results are demonstrated below:

<table>
    <tr>
        <th rowspan="2">Model</th><th colspan="6" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">1024</th><th align="center">2048</th><th align="center">4096</th><th align="center">8192</th><th align="center">16384</th><th align="center">32768</th>
    </tr>
     <tr>
        <td>Qwen-7B (original)</td><td align="center">4.23</td><td align="center">3.78</td><td align="center">39.35</td><td align="center">469.81</td><td align="center">2645.09</td><td align="center">-</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk</td><td align="center">4.23</td><td align="center">3.78</td><td align="center">3.59</td><td align="center">3.66</td><td align="center">5.71</td><td align="center">-</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn</td><td align="center">4.23</td><td align="center">3.78</td><td align="center">3.58</td><td align="center">3.56</td><td align="center">4.62</td><td align="center">-</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn + window_attn</td><td align="center">4.23</td><td align="center">3.78</td><td align="center">3.58</td><td align="center">3.49</td><td align="center">4.32</td><td align="center">-</td>
    </tr>
    <tr>
    <tr>
        <td>Qwen-1.8B</td><td align="center"><b>5.00</b></td><td align="center"><b>4.48</b></td><td align="center"><b>4.13</b></td><td align="center"><b>3.89</b></td><td align="center">17.42</td><td align="center">433.85</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn + window_attn</td><td align="center"><b>5.00</b></td><td align="center"><b>4.48</b></td><td align="center"><b>4.14</b></td><td align="center"><b>3.93</b></td><td align="center"><b>3.82</b></td><td align="center"><b>3.83</b></td>
    </tr>
    <tr>
        <td>Qwen-7B</td><td align="center"><b>4.23</b></td><td align="center"><b>3.81</b></td><td align="center"><b>3.52</b></td><td align="center"><b>3.31</b></td><td align="center">7.27</td><td align="center">181.49</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn + window_attn</td><td align="center"><b>4.23</b></td><td align="center"><b>3.81</b></td><td align="center"><b>3.52</b></td><td align="center"><b>3.33</b></td><td align="center"><b>3.22</b></td><td align="center"><b>3.17</b></td>
    </tr>
    <tr>
        <td>Qwen-14B</td><td align="center"><b>-</b></td><td align="center"><b>3.46</b></td><td align="center">22.79</td><td align="center">334.65</td><td align="center">3168.35</td><td align="center">-</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk + logn + window_attn</td><td align="center"><b>-</b></td><td align="center"><b>3.46</b></td><td align="center"><b>3.29</b></td><td align="center"><b>3.18</b></td><td align="center">3.42</td><td align="center">-</td>
    </tr>
    <tr>
        <td>Qwen-72B</td><td align="center"><b>-</b></td><td align="center"><b>-</b></td><td align="center">-</td><td align="center"><b>2.83</b></td><td align="center"><b>2.73</b></td><td align="center"><b>2.72</b></td>
    </tr>
    </tr>
</table>

Furthermore, to verify the ability of Qwen-72B-Chat on long text understanding, we tested it on [L-Eval](https://arxiv.org/abs/2307.11088) (closed-ended tasks). The results are as follows:

| Model             | Input Length | Average   |  Coursera  |    GSM     |   QuALITY  |    TOEFL   |   CodeU    |  SFcition  |
|:------------------|:------------:|:---------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ChatGPT-3.5-16k   |     16K      |   60.73   | **63.51**  | **84.00**  |   61.38    |    78.43   | **12.22**  |    64.84   |
| **Qwen-72B-Chat** |     32K      | **62.30** |   58.13    |   76.00    | **77.22**  |  **86.24** |    6.66    |  **69.53** |

We conducted the "needle in a haystack" experiment (the idea came from [@Greg Kamradt](https://twitter.com/GregKamradt/status/1727018183608193393)) to test whether the model can retrieve information at different positions in the inputs of different lengths, the result is as follows:

![](assets/qwen_72b_needle_in_a_haystack.png)

The above results show that Qwen-72B-Chat can accurately retrieve information placed in various positions within an input length of 32k, proving its excellent long text understanding capabilities.

## Tokenizer

Our tokenizer based on tiktoken is different from other tokenizers, e.g., sentencepiece tokenizer. You need to pay attention to special tokens, especially in finetuning. For more detailed information on the tokenizer and related use in fine-tuning, please refer to the [documentation](tokenization_note.md).
<br><br>

## Reproduction

For your reproduction of the model performance on benchmark datasets, we provide scripts for you to reproduce the results. Check [eval/EVALUATION.md](eval/EVALUATION.md) for more information. Note that the reproduction may lead to slight differences from our reported results.
<br><br>

## FAQ

If you meet problems, please refer to [FAQ](FAQ.md) and the issues first to search a solution before you launch a new issue.
<br><br>

## Citation
If you find our work helpful, feel free to give us a cite.

```
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```
<br>

## License Agreement

The source code provided at <https://github.com/QwenLM/Qwen> is licensed under the [Apache 2.0 License](./LICENSE) that can be found at the root directory.

Researchers and developers are free to use the codes and model weights of both Qwen and Qwen-Chat. For their commercial use, please check the License Agreement accompanying each model.

- Qwen-72B, Qwen-14B, and Qwen-7B are licensed under the [Tongyi Qianwen LICENSE AGREEMENT](./Tongyi%20Qianwen%20LICENSE%20AGREEMENT) that can be found at the corresponding HuggingFace and ModelScope repository. For commercial use, please fill out the form ([72B](https://dashscope.console.aliyun.com/openModelApply/Qwen-72B-Chat), [14B](https://dashscope.console.aliyun.com/openModelApply/Qwen-14B-Chat), and [7B](https://dashscope.console.aliyun.com/openModelApply/qianwen)) to apply.

- Qwen-1.8B is licensed under the [Tongyi Qianwen RESEARCH LICENSE AGREEMENT](./Tongyi%20Qianwen%20RESEARCH%20LICENSE%20AGREEMENT) that can be found at the corresponding HuggingFace and ModelScope repository. For commercial use, please contact us.
<br><br>

## Contact Us

If you are interested to leave a message to either our research team or product team, join our Discord or WeChat groups! Also, feel free to send an email to qianwen_opensource@alibabacloud.com.

