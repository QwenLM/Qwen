<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp ｜ &nbsp<a href="README_JA.md">日本語</a> ｜ &nbsp<a href="README_FR.md">Français</a>
</p>
<br><br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg" width="400"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2309.16609">Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-14B-Chat-Demo/summary">Demo</a>
<br>
<a href="assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp ｜ &nbsp&nbsp DingTalk (钉钉) &nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/z3GAxXZ9Ce">Discord</a>&nbsp&nbsp
</p>
<br><br>

|     |                                                              Qwen-Chat                                                               |                                                                Qwen-Chat (Int4)                                                                |                        Qwen-Chat (Int8)                         |                                                            Qwen                                                            |
|-----|:------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|
| 7B  |  <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>  |  <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int4/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int4">🤗</a>  | <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int8">🤗</a>  |  <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>  |
| 14B | <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-14B-Chat">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int4/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-14B-Chat-Int4">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int8/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-14B-Chat-Int8">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-14B/summary">🤖</a>  <a href="https://huggingface.co/Qwen/Qwen-14B">🤗</a> |



기본 언어(base language models)인 **Qwen**, **Qwen-7B**, **Qwen-14B**와 채팅 모델(chat models)인 **Qwen-Chat**을 포함한 **Qwen** 시리즈를 오픈소스화합니다. 링크는 위 표에 있으며, 해당 링크를 클릭하여 모델 카드를 확인하실 수 있습니다. 또한, [Qwen의 기술 보고서](https://arxiv.org/abs/2309.16609)를 공개합니다. 논문 링크를 클릭하여 확인해 주세요.

요약하자면, 당사는 광범위한 도메인과 언어(중국어와 영어를 중심으로) 등으로 구성 된 최대 3조 토큰의 다국어 데이터에 대해 안정적으로 사전 학습된 강력한 기본 언어 모델을 보유하고 있습니다. 벤치마크 데이터 세트에서 경쟁력 있는 성능을 달성하였으며, 채팅·콘텐츠 생성·정보 추출·요약·번역·코딩·수학 문제 풀이 등을 할 수 있습니다. 또한, 도구를 사용하거나 에이전트 역할을 할 수 있으며, 코드 인터프리터 역할 또한 할 수 있는 사람의 선호도에 맞춰진 SFT 및 RLHF(아직 출시되지 않음) 채팅 모델 역시 보유하고 있습니다.

이 레포지토리에서는 다음과 같은 내용을 소개합니다.

* Quick Start를 통한 Qwen을 이용한 간단한 추론 방법
* GPTQ 및 KV 캐시 양자화를 포함한 양자화 모델에 대한 세부 정보
* 속도와 메모리 비용을 포함한 추론 성능에 대한 통계
* 전체 파라미터 튜닝, LoRA 및 Q-LoRA를 포함한 미세 튜닝에 대한 튜토리얼
* 배포 가이드 (vLLM 및 FastChat의 예시 포함)
* WebUI, CLI 데모 등을 포함한 데모 구축 방법
* DashScope API 서비스 소개 및 모델에 대한 OpenAI 스타일 API 구축 가이드
* 도구 사용, 에이전트 및 코드 인터프리터를 위한 Qwen에 대한 정보
* 장문 이해도에 대한 평가 결과
* 라이센스
* 그 외 관련된 지식

문제가 발생한다면 [FAQ](FAQ.md)를 참조하여 도움을 받으실 수 있고, 추가적인 어려움이 있으시다면 언제든지 알려주십시요. (더 많은 사람과 문제를 공유하기 위해서 영어로 작성해주시면 감사하겠습니다.) 저희 팀을 돕고 싶으시다면, 주저하지 마시고 풀 리퀘스트를 보내주세요! 저희 팀은 항상 저희의 모델을 홍보하고자 노력하고 있습니다.

저희와 대화를 나누거나 커피 타임을 갖고 싶으시다면 디스코드 또는 위챗에 참여해주세요! 언제든지 환영합니다.
<br><br>

## News and Updates

* 2023.10.17 Int8 quantized model **Qwen-7B-Chat-Int8**, **Qwen-14B-Chat-Int8**을 릴리즈하였습니다.
* 2023.9.25 🔥 [qwen.cpp](https://github.com/QwenLM/qwen.cpp)와 [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)를 포함하여, ModelScope 및 Hugging Face에 **Qwen-14B**과 **Qwen-14B-Chat**를 릴리즈하였으며, Codes와 **Qwen-7B**, **Qwen-7B-Chat**모델들의 체크 포인트가 업데이트 되었습니다. **최신 버전을 Pull 해서 사용해주세요.**
    - 새로 업데이트 된 **Qwen-7B**는 이전 모델보다 더 많은 수의 훈련 토큰을 사용하였습니다. 훈련에 사용된 토큰의 수가 2.2T에서 2.4T로 증가했으며, 컨텍스트 길이는 2048에서 8192로 확장되었습니다. **Qwen-7B**의 중국어 지식과 코딩 능력이 더욱 향상되었습니다.
* 2023.9.12 전체 파라미터 미세 조정, LoRA 및 Q-LoRA를 포함한 Qwen-7B 모델에서 미세 조정을 지원합니다.
* 2023.8.21 낮은 메모리 비용으로 빠르게 추론 할 수 있는 Qwen-7B-Chat용 Int4 양자화 모델인 **Qwen-7B-Chat-Int4**를 출시합니다. 그럼에도 불구하고, 벤치마크 평가에서 성능 저하가 크지 않음을 보고합니다.
* 2023.8.3 모델스코프와 허깅 페이스에서 **Qwen-7B**와 **Qwen-7B-Chat**을 모두 출시합니다. 또한 훈련 세부 사항 및 모델 성능을 포함한 모델에 대한 자세한 내용은 기술 메모를 제공합니다.
<br>

## Performance

자연어 이해, 수학적 문제 해결, 코딩 등에 대한 모델의 능력을 평가하는 일련의 벤치마크 데이터 세트(예: MMLU, C-Eval, GSM8K, MATH, HumanEval, MBPP, BBH 등)에서 Qwen-14B와 Qwen-7B(더 많은 토큰으로 훈련된 새 버전이며 컨텍스트 길이가 2048에서 8192로 확장됨)는 유사한 모델 크기의 기준 모델보다 성능이 뛰어납니다. 그러나 Qwen-14B조차도 GPT-4는 말할 것도 없고 GPT-3.5에도 크게 뒤떨어집니다. 아래 결과를 참조하세요.

<p align="left">
    <img src="assets/radar_14b.jpg" width="600"/>
<p>
<br>

| Model              |   MMLU   |  C-Eval  |  GSM8K   |   MATH   | HumanEval |   MBPP   |   BBH    |  CMMLU   |
|:-------------------|:--------:|:--------:|:--------:|:--------:|:---------:|:--------:|:--------:|:--------:|
|                    |  5-shot  |  5-shot  |  8-shot  |  4-shot  |  0-shot   |  3-shot  |  3-shot  |  5-shot  |
| LLaMA2-7B          |   46.8   |   32.5   |   16.7   |   3.3    |   12.8    |   20.8   |   38.2   |   31.8   |
| LLaMA2-13B         |   55.0   |   41.4   |   29.6   |   5.0    |   18.9    |   30.3   |   45.6   |   38.4   |
| LLaMA2-34B         |   62.6   |    -     |   42.2   |   6.2    |   22.6    |   33.0   |   44.1   |    -     |
| ChatGLM2-6B        |   47.9   |   51.7   |   32.4   |   6.5    |     -     |    -     |   33.7   |    -     |
| InternLM-7B        |   51.0   |   53.4   |   31.2   |   6.3    |   10.4    |   14.0   |   37.0   |   51.8   |
| InternLM-20B       |   62.1   |   58.8   |   52.6   |   7.9    |   25.6    |   35.6   |   52.5   |   59.0   |
| Baichuan2-7B       |   54.7   |   56.3   |   24.6   |   5.6    |   18.3    |   24.2   |   41.6   |   57.1   |
| Baichuan2-13B      |   59.5   |   59.0   |   52.8   |   10.1   |   17.1    |   30.2   |   49.0   |   62.0   |
| Qwen-7B (original) |   56.7   |   59.6   |   51.6   |   10.4   |   24.4    |   31.2   |   40.6   |   58.8   |
| **Qwen-7B**        |   58.2   |   63.5   |   51.7   |   11.6   |   29.9    |   31.6   |   45.0   |   62.2   |
| **Qwen-14B**       | **66.3** | **72.1** | **61.3** | **24.8** | **32.3**  | **40.8** | **53.4** | **71.0** |


비교한 모든 모델에 대해 공식적으로 보고된 결과와 [OpenCompass](https://opencompass.org.cn/leaderboard-llm) 사이의 최고 점수를 보고합니다. 

더 많은 실험 결과(더 많은 벤치마크 데이터 세트에 대한 자세한 모델 성능) 및 자세한 내용은 [여기](https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf)를 클릭하여 기술 보고서를 참조하시기 바랍니다.

<br><br>

## Requirements

* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* transformers 4.32 and above
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
<br>

## Quickstart

아래에서는 🤖 모델스코프 및 🤗 트랜스포머와 함께 Qwen-Chat을 사용하는 방법을 보여주는 간단한 예제를 제공합니다.

코드를 실행하기 전에 환경을 설정하고 필요한 패키지를 설치했는지 확인하세요. 위의 요구 사항을 충족하는지 확인한 다음 종속 라이브러리를 설치하세요.

```bash
pip install -r requirements.txt
```

장치에서 fp16 또는 bf16을 지원하는 경우, 효율을 높이고 메모리 사용량을 줄이기 위해 [flash-attention](https://github.com/Dao-AILab/flash-attention)(**현재는 Flash Attention 2를 지원함**)을 설치하는 것을 권장합니다. (**플래시 어텐션은 선택 사항이며, 설치하지 않아도 프로젝트가 정상적으로 실행됩니다.**).

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# pip install csrc/rotary
```

이제 모델스코프 또는 트랜스포머로 사용하실 수 있습니다.

### 🤗 Transformers

추론에 Qwen-Chat을 사용하려면 아래에 설명된 대로 몇 줄의 코드를 입력하기만 하면 됩니다. "Qwen/Qwen-7B-Chat" 및 "Qwen/Qwen-14B-Chat"과 같이 올바른 모델 이름 또는 경로를 전달해야 하고, **최신 코드를 사용하셔야 합니다.**

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

사전 학습된 기본 모델을 실행하는 방법도 간단합니다.

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

HuggingFace에서 모델 체크포인트와 코드를 다운로드하는 동안 네트워크 문제가 발생하는 경우, 아래에 설명된 대로 모델스코프에서 체크포인트를 먼저 가져온 다음 로컬 디렉터리에서 로드하는 방법을 사용할 수 있습니다.

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

ModelScope는 서비스형 모델(MaaS)을 위한 오픈소스 플랫폼으로, AI 개발자에게 유연하고 비용 효율적인 모델 서비스를 제공합니다. 마찬가지로 아래와 같이 ModelScope로 모델을 실행할 수 있습니다.

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
Qwen은 일괄 추론을 지원합니다. 플래시 어텐션이 활성화된 상태에서 일괄 추론을 사용하면 속도가 40% 향상될 수 있습니다. 예제 코드는 아래와 같습니다.
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

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
CPU에 저희 모델을 배포하려면, Qwen과 tiktoken을 순수 C++로 구현한 [qwen.cpp](https://github.com/QwenLM/qwen.cpp)를 사용하실 것을 강력히 권장합니다. 자세한 내용은 저장소에서 확인하세요!

또한, CPU에서 직접 모델을 실행하는 것도 간단하지만 디바이스 사양이 필요합니다.

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
```

그러나 추론 효율성이 매우 낮을 가능성이 높습니다. (suffer from extremely low inference efficiency)

### Multiple GPUs

GPU 메모리가 부족하여 1개 이상의 GPU에서 모델을 실행하려는 경우, 이제 트랜스포머에서 지원하는 기본 로딩 방법을 직접 사용할 수 있습니다. utils.py`에 기반한 이전 방법은 더 이상 사용되지 않습니다.

그러나 이 방법은 간단하지만 네이티브 파이프라인 병렬 처리의 효율성이 낮습니다. FastChat과 함께 vLLM을 사용하는 것이 좋으며 배포 섹션을 읽어보시기 바랍니다.
<br><br>

## Quantization

### GPTQ

[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)를 기반으로 하는 솔루션을 제공하고 있으며, 무손실 모델 효과에 가깝지만 메모리 비용과 추론 속도 모두에서 성능이 향상된 Int4 양자화 모델을 출시했습니다.

여기에서는 제공된 양자화 모델을 추론에 사용하는 방법을 설명합니다. 시작하기 전에 자동-gptq의 요구 사항(예: 토치 2.0 이상, 트랜스포머 4.32.0 이상 등)을 충족하고 필요한 패키지를 설치했는지 확인하세요.

```bash
pip install auto-gptq optimum
```

만약 'auto-gptq' 설치에 문제가 있다면, 공식 [repo](https://github.com/PanQiWei/AutoGPTQ)에서 휠을 찾아보시길 권장합니다.

그러면 정량화된 모델을 쉽게 로드하고 평소와 동일하게 추론을 실행할 수 있습니다.

```python
# Model names: "Qwen/Qwen-7B-Chat-Int4", "Qwen/Qwen-14B-Chat-Int4"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "Hi", history=None)
```

벤치마크에서 BF16, Int8 및 Int4 모델의 모델 성능을 살펴본 결과, 양자화된 모델에서 성능 저하가 크지 않은 것으로 나타났습니다. 결과는 아래와 같습니다.

| Quantization         | MMLU | CEval (val) | GSM8K | Humaneval |
|----------------------|:----:|:-----------:|:-----:|:---------:|
| Qwen-7B-Chat (BF16)  | 55.8 |    59.7     | 50.3  |   37.2    |
| Qwen-7B-Chat (Int8)  | 55.4 |    59.4     | 48.3  |   34.8    |
| Qwen-7B-Chat (Int4)  | 55.1 |    59.2     | 49.7  |   29.9    |
| Qwen-14B-Chat (BF16) | 64.6 |    69.8     | 60.1  |   43.9    |
| Qwen-14B-Chat (Int8) | 63.6 |    68.6     | 60.0	 |   48.2    |
| Qwen-14B-Chat (Int4) | 63.3 |    69.0     | 59.8  |   45.7    |

### Quantization of KV cache

> 참고: 허깅 페이스의 내부 메커니즘으로 인해 이 기능에 대한 지원 파일
> (예: `cache_autogptq_cuda_256.cpp` 및 `cache_autogptq_cuda_kernel_245.cu`)이 누락될 수 있습니다. 수동으로 다운로드하세요.
> 수동으로 다운로드하여 다른 모듈 파일과 같은 폴더에 넣으세요.

주의 KV 캐시를 정량화하여 압축하여 저장하면 샘플 처리량을 높일 수 있습니다. 'use_cache_quantization' 및 'use_cache_kernel' 파라미터는 KV 캐시 양자화 동작을 제어하기 위해 제공됩니다.
`use_cache_quantization = True, use_cache_kernel = True`일 경우, kv-cache-quantization이 활성화됩니다.
구체적인 사용 방법은 다음과 같습니다.

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
Attention:
현재 kv-캐시 양자화 및 플래시 attn을 동시에 켤 수 없습니다.
kv 캐시 양자화와 use_flash_attn을 동시에 활성화하면(`use_flash_attn=True, use_cache_quantization=True, use_cache_kernel=True`), use_flash_attn은 기본적으로 비활성화됩니다(`use_flash_attn=false`).

정량화된 int8-kvcache 모델을 사용해도 다운스트림 평가에서 성능 저하가 크지 않음을 확인했습니다. 다운스트림 평가는 메모리 풋프린트에 중점을 두었습니다. 
프로파일링은 PyTorch 2.0.1 및 CUDA 11.4가 탑재된 단일 A100-SXM4-80G GPU에서 실행되었으며, BF16 모델을 사용하며 기본적으로 1024개의 토큰(seq-length=1024)을 생성하였고, oom은 메모리 부족을 의미합니다.

kv-캐시 양자화를 켜면 더 큰 배치 크기(bs)를 실행할 수 있습니다.

| USE KVCache |  bs=1  |  bs=4  | bs=16  | bs=32  | bs=64  | bs=100 |
|-------------|:------:|:------:|:------:|:------:|:------:|:------:|
| no          | 16.3GB | 24.1GB | 31.7GB | 48.7GB |  oom   |  oom   |
| yes         | 15.5GB | 17.2GB | 22.3GB | 30.2GB | 48.2GB | 72.4GB |

kv-캐시 양자화가 켜져 있으면 모델은 추론 시 더 긴 seq 길이(sl, 생성된 토큰 수)를 생성할 때 더 많은 메모리를 절약할 수 있습니다.

| USE KVCache | sl=512 | sl=1024 | sl=2048 | sl=4096 | sl=8192 |
|-------------|:------:|:-------:|:-------:|:-------:|:-------:|
| no          | 15.2GB | 16.3GB  | 17.6GB  | 19.5GB  | 23.2GB  |
| yes         |  15GB  | 15.5GB  | 15.8GB  | 16.6GB  | 17.6GB  |

kv-캐시 양자화를 켜는 모델은 레이어-패스트의 형식을 float에서 int8로 변환하고, 양자화된 레이어-패스트는 현재 값의 양자화 매개변수도 저장합니다.
구체적인 단계는 다음과 같습니다.

1、key와 value의 양자화 (Quantize key/value)
```
    qv,scale,zero_point=quantize_cache_v(v)
```
2、layer_past로 양자화된 key, value 저장 (Store into layer_past)

quantized layer_past의 포맷
```
    layer_past=((q_key,key_scale,key_zero_point),
                (q_value,value_scale,value_zero_point))
```
layer_past의 기본 포맷
```
    layer_past=(key,value)
```

다시 float으로 양자화된 attention KV를 사용하고자 할 경우, int8 key/value 값을 다음과 같이 float으로 역양자화(dequantization)하시면 됩니다.
```
    v=dequantize_cache_torch(qv,scale,zero_point)
```
<br>


## Inference Performance

이 섹션에서는 모델의 속도 및 메모리 통계를 다양한 정밀도로 제공합니다. 속도 및 메모리 프로파일링은 [이 스크립트](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py)를 사용하여 실행할 수 있습니다.

### Speed

플래시 주의 v1, v2를 사용하거나 사용하지 않는 조건에서 BF16, Int8, Int4의 정밀도로 모델을 사용하여 2048개와 8192개의 토큰을 생성할 때의 평균 추론 속도 (tokens/s)를 측정했습니다.

<table>
    <tr>
      <th rowspan="2">Model Size</th><th rowspan="2">Precision</th><th rowspan="2">FlashAttn</th><th colspan="2" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">2048</th><th align="center">8192</th>
    </tr>
    </tr>
    </tr>
    <tr>
        <th rowspan="9">7B</th><td align="center" rowspan="3">BF16</td><td align="center">v2</td><td align="center">40.93</td><td align="center">36.14</td>
    </tr>
    <tr>
        <td align="center">v1</td><td align="center">40.75</td><td align="center">35.34
    </tr>
    <tr>
        <td align="center">Disabled</td><td align="center">37.55</td><td align="center">33.56
    </tr>
    <tr>
        <td align="center" rowspan="3">Int8</td><td align="center">v2</td><td align="center">37.47</td><td align="center">32.54</td>
    </tr>
    <tr>
        <td align="center">v1</td><td align="center">37.51</td><td align="center">32.39
    </tr>
    <tr>
        <td align="center">Disabled</td><td align="center">37.84</td><td align="center">32.65
    </tr>
    <tr>
        <td align="center" rowspan="3">Int4</td><td align="center">v2</td><td align="center">50.09</td><td align="center">38.61</td>
    </tr>
    <tr>
        <td align="center">v1</td><td align="center">45.98</td><td align="center">36.47
    </tr>
    <tr>
        <td align="center">Disabled</td><td align="center">48.12</td><td align="center">36.70
    </tr>
    <tr>
        <th rowspan="9">14B</th><td align="center" rowspan="3">BF16</td><td align="center">v2</td><td align="center">32.88</td><td align="center">24.87</td>
    </tr>
    <tr>
        <td align="center">v1</td><td align="center">32.76</td><td align="center">28.89
    </tr>
    <tr>
        <td align="center">Disabled</td><td align="center">29.32</td><td align="center">22.91
    </tr>
    <tr>
        <td align="center" rowspan="3">Int8</td><td align="center">v2</td><td align="center">29.28</td><td align="center">24.22</td>
    </tr>
    <tr>
        <td align="center">v1</td><td align="center">28.31</td><td align="center">23.87
    </tr>
    <tr>
        <td align="center">Disabled</td><td align="center">31.12</td><td align="center">24.60
    </tr>
    <tr>
        <td align="center" rowspan="3">Int4</td><td align="center">v2</td><td align="center">38.72</td><td align="center">27.33</td>
    </tr>
    <tr>
        <td align="center">v1</td><td align="center">37.81</td><td align="center">26.46
    </tr>
    <tr>
        <td align="center">Disabled</td><td align="center">37.65</td><td align="center">26.00
    </tr>
</table>


구체적으로 프로파일링 설정은 2048개의 토큰을 인코딩하고 8192개의 새로운 토큰을 생성하는 것입니다. 프로파일링은 파이토치 2.0.1 및 CUDA 11.8이 설치된 단일 A100-SXM4-80G GPU에서 실행되었으며, 추론 속도는 인코딩 및 생성된 토큰에 대한 평균값으로 표시합니다.

참고: 위에 언급된 Int4/Int8 모델의 생성 속도는 autogptq 라이브러리를 사용합니다. ``AutoModelForCausalLM.from_pretrained``를 사용하여 로드된 모델의 속도는 현재 약 20% 느립니다. 이 문제는 허깅페이스 팀에 보고했으며, 해결 방안이 마련되는 즉시 업데이트하겠습니다.

### GPU Memory Usage

또한 2048개의 토큰을 컨텍스트로 인코딩하고 단일 토큰을 생성할 때와 (단일 토큰을 컨텍스트로 사용하여) 8192개의 토큰을 생성할 때, 각각 BF16, Int8 또는 Int4 양자화 수준에서 최대 GPU 메모리 사용량도 프로파일링했습니다. 결과(GB)는 아래와 같습니다.

<table>
    <tr>
      <th rowspan="2">Model Size</th><th rowspan="2">Precision</th><th colspan="2" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">2048</th><th align="center">8192</th>
    </tr>
    </tr>
    </tr>
    <tr>
        <th rowspan="3">7B</th><td align="center">BF16</td><td align="center">16.99</td><td align="center">22.53</td>
    </tr>
    <tr>
        <td align="center">Int8</td><td align="center">11.20</td><td align="center">16.62
    </tr>
    <tr>
        <td align="center">Int4</td><td align="center">8.21</td><td align="center">13.63</td>
    </tr>
    <tr>
        <th rowspan="3">14B</th><td align="center">BF16</td><td align="center">30.15</td><td align="center">38.94</td>
    </tr>
    <tr>
        <td align="center">Int8</td><td align="center">18.81</td><td align="center">27.54
    </tr>
    <tr>
        <td align="center">Int4</td><td align="center">13.01</td><td align="center">21.79</td>
    </tr>
</table>
<br>


## Finetuning

### Usage
이제 사용자가 다운스트림 애플리케이션을 위해 사전 학습된 모델을 간단한 방식으로 미세 조정할 수 있도록 공식 학습 스크립트인 `finetune.py`를 제공합니다. 또한 셸 스크립트를 제공하여 간편하게 미세 조정을 시작할 수 있습니다. [DeepSpped](https://github.com/microsoft/DeepSpeed) 및 [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/)를 통한 학습을 지원하며, 제공되는 셸 스크립트는 DeepSpeed(참고: 최신 버전의 pydantic과 충돌이 있을 수 있음)와 PEFT를 사용합니다. 설치 방법은 다음과 같습니다.

```bash
pip install peft deepspeed
```

학습 데이터를 준비하려면 모든 샘플을 목록에 넣고 json 파일에 저장해야 합니다. 각 샘플은 ID와 대화 목록으로 구성된 사전이며, 아래는 샘플 1개에 포함된 내용을 보여주는 간단한 예제입니다.
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

데이터 준비 후 제공된 셸 스크립트를 사용하여 미세 조정을 실행할 수 있습니다. 데이터 파일 경로인 `$DATA`를 지정하는 것을 잊지 마세요.

미세 조정 스크립트를 통해 다음을 수행할 수 있습니다.
- 전체 파라미터 미세 조정
- LoRA
- Q-LoRA

전체 파라미터 미세 조정을 수행하려면 전체 훈련 프로세스에서 모든 파라미터를 업데이트해야 합니다. 트레이닝을 시작하려면 다음 스크립트를 실행하세요.

```bash
# Distributed training. We do not provide single-GPU training script as the insufficient GPU memory will break down the training.
sh finetune/finetune_ds.sh
```

셸 스크립트에서 올바른 모델 이름 또는 경로, 데이터 경로, 출력 디렉터리를 지정해야만 하며, 이 스크립트는 DeepSpeed ZeRO3를 사용한다는 것을 주의하세요. 변경을 원하시면 `--deepspeed` 인수를 제거하거나 요구 사항에 따라 DeepSpeed config json 파일을 변경하시면 됩니다. 또한, 이 스크립트는 혼합 정밀도 훈련을 지원하므로, 사용을 원하실 경우 `--bf16 True` 또는 `--fp16 True`인자를 전달하세요. 혼합 정밀도 훈련(mixed-precision training)으로 인해 fp16을 사용할 때는, 반드시 DeepSpeed를 사용해야 한다는 점을 기억하세요. 
경험적으로 사용 중인 머신이 bf16을 지원하는 경우 사전 훈련 및 정렬과 일관된 훈련을 위해 bf16을 사용하는 것이 좋으며, 따라서 기본값으로 설정해두었습니다.

마찬가지로 LoRA를 실행하려면 아래와 같이 다른 스크립트를 사용하여 실행합니다. 시작하기 전에 `PEFT`를 설치했는지 확인하시고, 모델·데이터·출력에 대한 경로를 지정하세요. LoRA는 어댑터만 저장하고, 어댑터를 구성하는 config json 파일의 경로는 로드할 사전 학습 모델을 찾는 데 사용되기 때문에 사전 학습 모델을 지정하실 때, 상대 경로보다 절대 경로를 사용하는 것을 추천합니다. 이 스크립트는 bf16과 fp16을 모두 지원합니다.

```bash
# Single GPU training
sh finetune/finetune_lora_single_gpu.sh
# Distributed training
sh finetune/finetune_lora_ds.sh
```

전체 매개변수 미세 조정과 비교할 때 LoRA([논문 링크](https://arxiv.org/abs/2106.09685))는 어댑터 레이어의 매개변수만 업데이트하고, 원래의 LLM의 레이어는 고정된 상태로 유지됩니다. 따라서 메모리 비용이 훨씬 적게 들고 계산 비용도 적게 듭니다. 

기본 언어 모델에 ChatML 형식에서 가져온 특수 토큰에 대한 지식이 없기 때문에 LoRA를 사용하여 채팅 모델(예: Qwen-7B-Chat) 대신 기본 언어 모델(예: Qwen-7B)을 미세 조정하려는 경우 스크립트가 임베딩 및 출력 레이어를 학습 가능한 파라미터로 자동 전환한다는 점에 유의하세요. 따라서 모델이 토큰을 이해하고 예측하려면 임베딩 및 출력 레이어를 업데이트해야만 합니다. 다시 말해, 학습을 통해 LoRA에서 특수 토큰을 가져오는 경우 코드 내에서 `modules_to_save`를 설정하여 레이어를 학습 가능한 파라미터 형태로 다시 설정해줘야 합니다. 또한, 이러한 파라미터를 훈련할 수 있는 경우 ZeRO3를 사용할 수 없으므로 스크립트에서 기본적으로 ZeRO2를 사용하게 됩니다. 새로 학습할 수 있는 파라미터가 없는 경우 DeepSpeed 구성 파일을 변경하여 ZeRO3로 전환하실 수 있으며, 이러한 학습 가능한 파라미터가 있는 경우와 없는 경우 LoRA의 메모리 사용량에 상당한 차이가 있음을 확인했습니다. 따라서 메모리 문제가 있는 경우 LoRA에서 채팅 모델을 미세 조정하는 것이 좋습니다. 자세한 내용은 아래 내용을 확인하세요. 

만약 LoRA 등을 사용해도 메모리가 부족하다면 양자화된 대규모 언어 모델과 페이징 주의와 같은 기타 기법을 사용하여 메모리 비용을 훨씬 더 적게 사용할 수 있는 Q-LoRA([논문 링크](https://arxiv.org/abs/2305.14314))를 고려해 볼 수 있습니다. 

참고: 단일 GPU Q-LoRA 트레이닝을 실행하려면 `pip` 또는 `conda`를 통해 `mpi4py`를 설치해야 할 수 있습니다.

Q-LoRA를 실행하려면 다음 스크립트를 직접 실행하세요.

```bash
# Single GPU training
sh finetune/finetune_qlora_single_gpu.sh
# Distributed training
sh finetune/finetune_qlora_ds.sh
```

Q-LoRA의 경우, 제공된 정량화된 모델(예: Qwen-7B-Chat-Int4)을 로드할 것을 권장합니다. 전체 파라미터 미세 조정 및 LoRA와 달리 **Q-LoRA에는 fp16만 지원하기 때문에 bf16 모델을 사용해서는 안됩니다.** 전체 파라미터 미세 조정 및 LoRA와 달리 Q-LoRA에는 fp16만 지원됩니다. 단일 GPU 훈련의 경우 torch amp로 인한 오류가 발생하기 때문에 혼합 정밀도 훈련에는 deepspeed를 사용해야 합니다. 또한 Q-LoRA의 경우에도 위에서 언급한 LoRA의 특수 토큰에 대한 문제가 여전히 존재합니다. 하지만 저희는 채팅 모델에 Int4 모델만 제공하고, 일반 모델의 경우 ChatML 형식의 특수 토큰을 이미 학습했기 때문에 레이어에 대한 걱정은 하지 않아도 됩니다. 단, Int4 모델의 레이어는 학습이 불가능하므로 학습에 특수 토큰을 도입하면 Q-LoRA가 작동하지 않을 수 있습니다.

> 참고: 허깅 페이스의 내부 메커니즘으로 인해 특정 파이썬이 아닌 파일(예: `*.cpp` 및 `*.cu`) 
> 들이 체크포인트에서 누락되었을 수 있습니다. 다른 파일이 포함된 디렉터리에 수동으로 복사해서 사용해야될 수 있습니다.

전체 매개변수 미세 조정과 달리 LoRA 및 Q-LoRA를 통한 학습은 어댑터의 매개변수만을 저장합니다. Qwen-7B에서 학습이 시작된다고 가정하면 다음과 같이 추론을 위한 미세 조정된 모델을 로드할 수 있습니다.

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()
```

어댑터를 병합하고 미세 조정된 모델을 독립형 모델로 저장하려면 다음 코드를 실행하면 됩니다. (이 작업은 LoRA에서만 가능하며 Q-LoRA에서 파라미터를 병합할 수 없음)
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

`new_model_directory` 폴더에는 병합된 모델 가중치와 모듈 파일이 포함되고, 저장된 파일에 `*.cu` 및 `*.cpp` 파일이 누락될 수 있음을 유의하시기 바랍니다. KV 캐시 기능을 사용하려면 해당 파일을 수동으로 복사하시기 바랍니다. 또한, 이 단계에서 토큰화기 파일은 새 디렉토리에 저장되지 않으므로, tokenizer files을 복사하거나 다음 코드를 사용하세요.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)

tokenizer.save_pretrained(new_model_directory)
```


참고: 멀티 GPU 트레이닝의 경우, 머신에 따라 분산 트레이닝에 적합한 하이퍼파라미터를 지정해야 하고, 데이터·메모리 사용량·훈련 속도 등을 고려하여  `--model_max_length` 인자를 사용하여 최대 시퀀스 길이를 지정하는 것이 좋습니다.

### Profiling of Memory and Speed
단일 GPU 트레이닝 설정에서 LoRA와 Q-LoRA의 GPU 메모리와 트레이닝 속도를 프로파일링합니다. (LoRA(emb)는 임베딩 및 출력 레이어를 학습하는 것을 의미하며, LoRA에는 학습 가능한 임베딩 및 출력 레이어가 없음을 유의) 이 테스트에서는 단일 A100-SXM4-80G GPU에서 실험하고, CUDA 11.8 및 Pytorch 2.0을 사용합니다. Flash Attention-2가 적용되고, 배치 크기 1과 그라데이션 누적 8을 균일하게 사용하여 256, 512, 1024, 2048, 4096, 8192 등 다양한 길이의 입력에 대한 메모리(GB)와 속도(s/iter)를 프로파일링합니다. 또한, 2개의 A100 GPU에서 Qwen-7B를 사용한 전체 파라미터 미세 조정에 대한 통계도 제공합니다. GPU 메모리의 제한으로 인해 256, 512, 1024 토큰의 통계는 다음과 같습니다.

<table>
    <tr>
      <th rowspan="2">Model Size</th><th rowspan="2">Method</th><th colspan="6" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">256</th><th align="center">512</th><th align="center">1024</th><th align="center">2048</th><th align="center">4096</th><th align="center">8192</th>
    </tr>
    </tr>
    </tr>
    <tr>
        <th rowspan="4">7B</th><td>LoRA</td><td align="center">20.1G / 1.2s/it</td><td align="center">20.4G / 1.5s/it</td><td align="center">21.5G / 2.8s/it</td><td align="center">23.8G / 5.2s/it</td><td align="center">29.7G / 10.1s/it</td><td align="center">36.6G / 21.3s/it</td>
    </tr>
    <tr>
        <td>LoRA (emb)</td><td align="center">33.7G / 1.4s/it</td><td align="center">34.1G / 1.6s/it</td><td align="center">35.2G / 2.9s/it</td><td align="center">35.1G / 5.3s/it</td><td align="center">39.2G / 10.3s/it</td><td align="center">48.5G / 21.7s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td><td align="center">11.5G / 3.0s/it</td><td align="center">11.5G / 3.0s/it</td><td align="center">12.3G / 3.5s/it</td><td align="center">13.9G / 7.0s/it</td><td align="center">16.9G / 11.6s/it</td><td align="center">23.5G / 22.3s/it</td>
    </tr>
    <tr>
        <td>Full-parameter</td><td align="center">139.2G / 4.0s/it</td><td align="center">148.0G / 4.0s/it</td><td align="center">162.0G / 4.5s/it</td><td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
        <th rowspan="3">14B</th><td>LoRA</td><td align="center">34.6G / 1.6s/it</td><td align="center">35.1G / 2.4s/it</td><td align="center">35.3G / 4.4s/it</td><td align="center">37.4G / 8.4s/it</td><td align="center">42.5G / 17.0s/it</td><td align="center">55.2G / 36.0s/it</td>
    </tr>
    <tr>
        <td>LoRA (emb)</td><td align="center">51.2 / 1.7s/it</td><td align="center">51.1G / 2.6s/it</td><td align="center">51.5G / 4.6s/it</td><td align="center">54.1G / 8.6s/it</td><td align="center">56.8G / 17.2s/it</td><td align="center">67.7G / 36.3s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td><td align="center">18.7G / 5.3s/it</td><td align="center">18.4G / 6.3s/it</td><td align="center">18.9G / 8.2s/it</td><td align="center">19.9G / 11.8s/it</td><td align="center">23.0G / 20.1s/it</td><td align="center">27.9G / 38.3s/it</td>
    </tr>
</table>
<br>

## Deployment

### vLLM 
배포 및 빠른 추론을 위해서는 FastChat과 함께 vLLM을 사용하는 것이 좋습니다. 먼저 패키지를 설치합니다.
```bash
pip install vllm
pip install "fschat[model_worker,webui]"
```
또는 `git clone` 및 `pip install -e .`를 사용하여 소스에서 설치할 수 있습니다. 설치 시 문제가 발생하면 해당 문서를 읽어보시기 바랍니다. 

vLLM 및 FastChat과 함께 Qwen을 실행하려면 먼저 컨트롤러를 실행해야 합니다.
```bash
python -m fastchat.serve.controller
```

그런 다음 모델 워커를 시작하면 추론을 위해 모델을 로드할 수 있습니다. 단일 GPU 추론의 경우 직접 실행할 수 있습니다.
```bash
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code
```
그러나 더 빠른 추론이나 더 큰 메모리를 위해 여러 GPU에서 모델을 실행하려는 경우 vLLM에서 지원하는 병렬 처리 텐서를 사용할 수 있습니다. 4개의 GPU에서 모델을 실행한다고 가정하면, 다음과 같은 명령어를 사용할 수 있습니다.
```bash
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --tensor-parallel-size 4
```

모델 워커를 실행한 후 원하는 대로 웹 데모 또는 OpenAI API를 실행할 수 있습니다. 웹 데모의 경우 다음 명령을 실행합니다.
```bash
python -m fastchat.serve.gradio_web_server
```
OpenAI API의 경우, 먼저 설치에 대한 OpenAI API 설명서를 확인하세요. 그런 다음 명령을 실행합니다.
```bash
python -m fastchat.serve.openai_api_server --host localhost --port 8000
```
<br>

## Demo

### Web UI

트위터에서는 사용자가 웹 UI 데모를 구축할 수 있는 코드를 제공(@wysaid)합니다. 시작하기 전에 다음 패키지를 설치해야 합니다.

```
pip install -r requirements_web_demo.txt
```

그런 다음 아래 명령을 실행하고 생성된 링크를 클릭합니다.

```bash
python web_demo.py
```

<p align="center">
    <br>
    <img src="assets/web_demo.gif" width="600" />
    <br>
<p>

### CLI Demo

생성에 대한 스트리밍 출력을 지원하는 CLI 데모 예제를 `cli_demo.py`에서 제공합니다. 사용자는 프롬프트를 입력하여 Qwen-7B-Chat과 상호 작용할 수 있으며, 모델은 스트리밍 모드에서 모델 출력을 반환합니다. 아래 명령어를 실행합니다.

```bash
python cli_demo.py
```

<p align="center">
    <br>
    <img src="assets/cli_demo.gif" width="600" />
    <br>
<p>
<br>

## API

API를 통해 Qwen을 사용하는 가장 간단한 방법은 알리바바 클라우드를 통한 DashScope API 서비스로 다음과 같이 사용하실 수 있습니다. 또한, 자체 서버에 OpenAI 스타일 API를 배포할 수 있는 스크립트도 제공합니다.

### DashScope
DashScope는 알리바바 클라우드에서 제공하는 대규모 언어 모델 API 서비스로, 이제 Qwen을 지원합니다. DashScope의 모델은 현재 세부 정보가 제공되지 않는 사내 버전입니다. 이 서비스에는 'qwen-turbo'와 'qwen-plus'가 있으며, 전자는 더 빠르게 실행되고 후자는 더 나은 성능을 달성합니다. 자세한 내용은 [공식 문서](https://dashscope.aliyun.com)를 참조하세요.

[공식 웹사이트](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.6c2774fahtfXdn)에서 대시스코프 계정을 생성하고 API 키(AK)를 발급받으시고, 환경 변수로 AK를 설정하시는 것을 권장합니다.
```bash
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
```
그런 다음 패키지를 설치하고 [다음 문서](https://help.aliyun.com/zh/dashscope/developer-reference/install-dashscope-sdk)를 확인하세요. Python을 사용하는 경우, pip로 DashScope를 설치할 수 있습니다.
```bash
pip install dashscope
```
JAVA SDK를 사용하는 경우 이 방법으로 설치할 수 있습니다.
```xml
<!-- https://mvnrepository.com/artifact/com.alibaba/dashscope-sdk-java -->
<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>dashscope-sdk-java</artifactId>
    <version>the-latest-version</version>
</dependency>
```
DashScope를 사용하는 가장 간단한 방법은 OpenAI API와 유사한 메시지와 함께 사용하는 것입니다. 아래에 예시가 나와 있습니다.
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
자세한 사용법은 공식 웹사이트에서 확인하세요.

### OpenAI API

OpenAI API를 기반으로 로컬 API를 배포하는 방법을 제공(@hanpenggit)합니다. 시작하기 전에 필요한 패키지를 설치하세요.

```bash
pip install fastapi uvicorn openai "pydantic>=2.3.0" sse_starlette
```

그런 다음 명령을 실행하여 API를 배포합니다:
```bash
python openai_api.py
```

인수를 변경할 수 있습니다(예: 체크포인트 이름 또는 경로의 경우 `-c`, CPU 배포의 경우 `--cpu-only` 등). API 배포를 실행하는 데 문제가 있는 경우, 패키지를 최신 버전으로 업데이트하면 문제를 해결할 수 있습니다.

API 사용 방법도 간단합니다. 아래 예시를 참조하세요.

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

**함수 호출**도 지원됩니다. 현재는 `stream=False`일 때만 지원하며, [사용 예시](examples/function_call_examples.py)를 참조하세요.
<br><br>



## Tool Usage

Qwen-Chat은 도구 사용과 함수 호출 기능에 최적화되어 있습니다. 사용자는 에이전트, LangChain 애플리케이션을 개발할 수 있으며, 파이썬 코드 인터프리터로 Qwen을 보강할 수도 있습니다.

리액트 프롬프트 원칙에 따라 도구 호출을 구현하는 방법에 대한 설명서는 [리액트 예제](examples/react_prompt.md)를 참조하세요. 이 원칙에 따라 [openai_api.py](openai_api.py)에서 함수 호출을 지원하고 있습니다.

중국 오픈소스 평가 벤치마크에서 모델의 도구 호출 기능을 테스트한 결과, Qwen-Chat은 일관되게 우수한 성능을 보였습니다.

<table>
    <tr>
        <th colspan="4" align="center">Chinese Tool-Use Benchmark</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Tool Selection (Acc.↑)</th><th align="center">Tool Input (Rouge-L↑)</th><th align="center">False Positive Error↓</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">95%</td><td align="center">0.90</td><td align="center">15.0%</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">85%</td><td align="center">0.88</td><td align="center">75.0%</td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td><td align="center">98%</td><td align="center">0.91</td><td align="center">7.3%</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">98%</td><td align="center">0.93</td><td align="center">2.4%</td>
    </tr>
</table>

수학 문제 해결, 데이터 시각화, 파일 처리 및 웹 스크래핑과 같은 기타 범용 작업에 Python 코드 인터프리터를 사용하는 Qwen의 능력을 평가하기 위해 이러한 기능을 평가하기 위해 특별히 고안된 벤치마크를 만들어 오픈 소스화했습니다. 벤치마크는 이 [링크](https://github.com/QwenLM/Qwen-Agent/tree/main/benchmark)에서 확인할 수 있습니다.

Qwen은 코드 생성 시 코드 실행성과 결과 정확도 측면에서 우수한 성능을 보였습니다.

<table>
    <tr>
        <th colspan="4" align="center">Executable Rate of Generated Code (%)</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Math↑</th><th align="center">Visualization↑</th><th align="center">General↑</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">91.9</td><td align="center">85.9</td><td align="center">82.8</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">89.2</td><td align="center">65.0</td><td align="center">74.1</td>
    </tr>
    <tr>
        <td>LLaMA2-7B-Chat</td>
        <td align="center">41.9</td>
        <td align="center">33.1</td>
        <td align="center">24.1 </td>
    </tr>
    <tr>
        <td>LLaMA2-13B-Chat</td>
        <td align="center">50.0</td>
        <td align="center">40.5</td>
        <td align="center">48.3 </td>
    </tr>
    <tr>
        <td>CodeLLaMA-7B-Instruct</td>
        <td align="center">85.1</td>
        <td align="center">54.0</td>
        <td align="center">70.7 </td>
    </tr>
    <tr>
        <td>CodeLLaMA-13B-Instruct</td>
        <td align="center">93.2</td>
        <td align="center">55.8</td>
        <td align="center">74.1 </td>
    </tr>
    <tr>
        <td>InternLM-7B-Chat-v1.1</td>
        <td align="center">78.4</td>
        <td align="center">44.2</td>
        <td align="center">62.1 </td>
    </tr>
    <tr>
        <td>InternLM-20B-Chat</td>
        <td align="center">70.3</td>
        <td align="center">44.2</td>
        <td align="center">65.5 </td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td>
        <td align="center">82.4</td>
        <td align="center">64.4</td>
        <td align="center">67.2 </td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td>
        <td align="center">89.2</td>
        <td align="center">84.1</td>
        <td align="center">65.5</td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="4" align="center">Accuracy of Code Execution Results (%)</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Math↑</th><th align="center">Visualization-Hard↑</th><th align="center">Visualization-Easy↑</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">82.8</td><td align="center">66.7</td><td align="center">60.8</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">47.3</td><td align="center">33.3</td><td align="center">55.7</td>
    </tr>
    <tr>
        <td>LLaMA2-7B-Chat</td>
        <td align="center">3.9</td>
        <td align="center">14.3</td>
        <td align="center">39.2 </td>
    </tr>
    <tr>
        <td>LLaMA2-13B-Chat</td>
        <td align="center">8.3</td>
        <td align="center">8.3</td>
        <td align="center">40.5 </td>
    </tr>
    <tr>
        <td>CodeLLaMA-7B-Instruct</td>
        <td align="center">14.3</td>
        <td align="center">26.2</td>
        <td align="center">60.8 </td>
    </tr>
    <tr>
        <td>CodeLLaMA-13B-Instruct</td>
        <td align="center">28.2</td>
        <td align="center">27.4</td>
        <td align="center">62.0 </td>
    </tr>
    <tr>
        <td>InternLM-7B-Chat-v1.1</td>
        <td align="center">28.5</td>
        <td align="center">4.8</td>
        <td align="center">40.5 </td>
    </tr>
    <tr>
        <td>InternLM-20B-Chat</td>
        <td align="center">34.6</td>
        <td align="center">21.4</td>
        <td align="center">45.6 </td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td>
        <td align="center">41.9</td>
        <td align="center">40.5</td>
        <td align="center">54.4 </td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td>
        <td align="center">58.4</td>
        <td align="center">53.6</td>
        <td align="center">59.5</td>
    </tr>
</table>

<p align="center">
    <br>
    <img src="assets/code_interpreter_showcase_001.jpg" />
    <br>
<p>

또한, 저희 모델이 허깅페이스 에이전트 역할을 할 수 있음을 입증하는 실험 결과도 제공합니다. 자세한 내용은 [예제 문서](examples/transformers_agent.md)를 참조하시기 바랍니다. 허깅페이스에서 제공하는 평가 데이터 세트에 대한 모델의 성능은 다음과 같습니다.

<table>
    <tr>
        <th colspan="4" align="center">HuggingFace Agent Benchmark- Run Mode</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Tool Selection↑</th><th align="center">Tool Used↑</th><th align="center">Code↑</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">100</td><td align="center">100</td><td align="center">97.4</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">95.4</td><td align="center">96.3</td><td align="center">87.0</td>
    </tr>
    <tr>
        <td>StarCoder-Base-15B</td><td align="center">86.1</td><td align="center">87.0</td><td align="center">68.9</td>
    </tr>
    <tr>
        <td>StarCoder-15B</td><td align="center">87.0</td><td align="center">88.0</td><td align="center">68.9</td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td><td align="center">87.0</td><td align="center">87.0</td><td align="center">71.5</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">93.5</td><td align="center">94.4</td><td align="center">87.0</td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="4" align="center">HuggingFace Agent Benchmark - Chat Mode</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Tool Selection↑</th><th align="center">Tool Used↑</th><th align="center">Code↑</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">98.5</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">97.3</td><td align="center">96.8</td><td align="center">89.6</td>
    </tr>
    <tr>
        <td>StarCoder-Base-15B</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">91.1</td>
    </tr>
    <tr>
        <td>StarCoder-15B</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">89.6</td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td><td align="center">94.7</td><td align="center">94.7</td><td align="center">85.1</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">95.5</td>
    </tr>
</table>

<br>

## Long-Context Understanding
컨텍스트 길이를 확장하고 훈련 시퀀스 길이의 병목 현상을 해소하기 위해 NTK 인식 보간, 윈도우 주의, LogN 주의 스케일링 등 여러 기법을 도입하여 Qwen-7B/14B의 컨텍스트 길이를 2k에서 8k 토큰 이상으로, Qwen-7B는 8k에서 32k 토큰으로 확장합니다. PPL 평가와 함께 arXiv 데이터 세트에서 언어 모델링 실험을 수행한 결과, 긴 컨텍스트 시나리오에서 Qwen이 뛰어난 성능을 발휘할 수 있음을 확인했습니다. 결과는 다음과 같습니다.

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
</table>


## Tokenizer

tiktoken 기반의 tokenizer는 문장 단위 tokenizer와 같은 다른 tokenizer와 다릅니다. 특히 미세 조정 시 특수 토큰(special tokens)을 제대로 처리해야합니다. tokenizer에 대한 자세한 설명과 미세 조정 관련 사용법은 [tokenizer 관련 문서](tokenization_note.md)를 참고하시기 바랍니다.

<br><br>

## Reproduction

벤치마크 데이터 세트에서 모델 성능을 재현할 수 있도록 결과를 재현할 수 있는 스크립트를 제공합니다. 자세한 내용은 [eval/EVALUATION.md](eval/EVALUATION.md)를 확인하세요. 재현 결과는 보고된 결과와 약간의 차이가 있을 수도 있습니다.
<br><br>

## FAQ

문제가 발생하면 새 이슈를 시작하기 전에 먼저 [자주 묻는 질문](FAQ.md)과 이슈를 참조하여 해결 방법을 찾아보시기 바랍니다.
<br><br>

## Citation
도움이 되셨다면, 다음을 자유롭게 인용해주세요.

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

연구자와 개발자는 Qwen과 Qwen-Chat의 코드와 모델 가중치를 자유롭게 사용할 수 있으며, 상업적 사용도 허용됩니다. 자세한 내용은 [LICENSE](LICENSE)에서 확인하세요. 상업적 사용에 대한 요구 사항이 있는 경우 양식 ([7B](https://dashscope.console.aliyun.com/openModelApply/qianwen), [14B](https://dashscope.console.aliyun.com/openModelApply/Qwen-14B-Chat))을 작성하여 신청하시기 바랍니다.
<br><br>

## Contact Us

저희 리서치팀이나 제품팀에 메시지를 남기고 싶으시다면, Discord 또는 WeChat 그룹에 가입하시거나 qianwen_opensource@alibabacloud.com로 이메일을 남겨주세요.