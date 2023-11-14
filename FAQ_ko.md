# FAQ

## Installation & Environment

#### Flash Attention 설치 실패

Flash Attention는 훈련 및 추론을 가속화하기 위한 옵션으로, H100, A100, RTX 3090, T4, RTX 2080과 같은  Turing, Ampere, Ada, and Hopper 아키텍처의 NVIDIA GPU만 Flash Attention을 사용할 수 있습니다. 그러나 **설치하지 않고도 당사 모델을 사용할 수 있습니다.**

#### 어떤 버전의 트랜스포머를 사용해야 하나요?

4.32.0 버전을 권장합니다.

#### 코드와 체크포인트를 다운로드했지만 모델을 로컬로 로드할 수 없습니다. 어떻게 해야 하나요?

코드를 최신으로 업데이트하고 모든 샤딩된 체크포인트 파일을 올바르게 다운로드했는지 확인하세요.

#### `qwen.tiktoken` is not found. qwen.tiktoken이 뭔가요?

tokenizer의 병합 파일로 다운로드하셔야 합니다. [git-lfs](https://git-lfs.com ) 없이 레포지토리만 클론하면 이 파일을 다운로드할 수 없습니다.

#### transformers_stream_generator/tiktoken/accelerate not found 오류

명령어 'pip install -r requirements.txt'를 실행합니다. 파일은 [https://github.com/QwenLM/Qwen-7B/blob/main/requirements.txt ](https://github.com/QwenLM/Qwen/blob/main/requirements.txt) 에서 확인할 수 있습니다.
<br><br>

## Demo & Inference

#### 데모가 있나요? CLI 데모와 웹 UI 데모가 있나요?

네, 웹 데모는 web_demo.py, CLI 데모는 cli_demo.py를 참조하십시오. 자세한 내용은 README를 참조하십시오.


#### CPU만 사용할 수 있나요?

네, 'python cli_demo.py --cpu-only'를 실행하면 모델과 추론이 CPU에만 로드됩니다.

#### 스트리밍을 지원하나요?

네, modeling_qwen.py의 chat_stream 함수를 참조하십시오.

#### chat_stream()을 사용하면, 모델이 횡설수설합니다.

토큰은 바이트를 나타내며 단일 토큰은 무의미한 문자열일 수 있기 때문입니다. 이러한 디코딩 결과를 방지하기 위해 tokenizer의 기본 설정을 업데이트했습니다. 코드를 최신 버전으로 업데이트하십시오.

#### 생성된 내용이 프롬프트의 지시 내용과 관련이 없는 것 같아요.

Qwen 대신 Qwen-Chat을 로딩하는지 확인 부탁드립니다. Qwen은 정렬되지 않은 기본 모델로 SFT/Chat 모델과 다르게 동작합니다.

#### quantization 지원 하나요?

네, quantization는 AutoGPTQ에서 지원합니다.


#### 긴 시퀀스를 처리할 때 속도가 느려요.

코드를 최신 버전으로 업데이트하면 도움이 됩니다.

#### 긴 시퀀스를 처리하는 과정에서 만족스럽지 못한 성능을 보여요.

NTK가 적용되었는지 확인하십시오. config.json의 use_dynamc_ntk 및 use_logn_attn은 true(기본값은 true)로 설정해야 합니다.
<br><br>



## Finetuning


#### Qwen은 SFT 또는 RLHF까지 지원하나요?

네, 이제 Full-parameter finetuning, LoRA, Q-LoRA 등 SFT를 지원합니다. 또한 [FastChat](**[https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat)), [Firefly]([https://github.com/yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)), [**LLaMA Efficient Tuning**]([https://github.com/hiyouga/LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning))도 사용할 수 있습니다.

그러나 현재는 RLHF를 지원하지 않습니다. 가까운 시일 내에 코드를 제공하도록 하겠습니다.
<br><br>


## Tokenizer

#### bos_id/eos_id/pad_id not found 오류

교육에서는 '<|endoftext|>'만 구분자 및 패딩 토큰으로 사용합니다. bos_id, eos_id, pad_id를 tokenizer.eod_id로 설정할 수 있습니다. tokenizer에 대한 자세한 내용은 tokenizer 관련 문서를 통해 확인하실 수 있습니다.