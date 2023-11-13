# 토큰화, Tokenization

Qwen-7B는 tiktoken 패키지를 사용하여, UTF-8 바이트에 BPE tokenization을 적용합니다.
Qwen-7B에는 BPE의 일반 토큰(regular tokens of type `bytes`)과 특수/제어 토큰(special/control tokens of type `str`), 두 종류의 토큰이 존재합니다. *(역자: 상세한 설명을 위해 일부 의역이 있으므로, 원문과 비교하며 읽어주세요.)*

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)
```

## 일반 토큰, Regular tokens

일반 토큰은 UTF-8 인코딩을 사용하여, 텍스트의 바이트 시퀀스로부터 학습된 BPE 토큰으로부터 모든 텍스트를 토큰화할 수 있도록 하며, 알려지지 않은 토큰은 존재하지 않습니다. 그러나 매우 드물게 등장하는 텍스트를 토큰화할 경우, 단일 바이트를 사용하기도 합니다. 따라서 이 과정에서 UTF-8 디코딩 오류가 발생할 수 있고, 디코딩 오류가 발생할 경우 기본적으로 `대체(replace)`되므로, 불완전한 생성으로 대체 된 문자(�)가 나타날 수도 있습니다.
이러한 오류를 수정하시고 싶으면, `decode` 함수에 `errors="ignore"` 인자를 일시적으로 전달하거나 `from_pretrained` 함수에 영구적으로 전달하시면 됩니다.
`errors`의 더 많은 옵션을 보시려면 다음 [파이썬 문서](https://docs.python.org/3/library/stdtypes.html#bytes.decode)를 참조하세요.

```python
>>> tokenizer.decode([51461])
' �'

>>> tokenizer.convert_ids_to_tokens([51461])
[b' \xe6\xa0']

>>> b' \xe6\xa0'.decode("utf-8", errors='replace')
' �'

>>> tokenizer.decode([51461, 117])
' 根'

>>> tokenizer.convert_ids_to_tokens([51461, 117])
[b' \xe6\xa0', b'\xb9']

>>> b' \xe6\xa0\xb9'.decode("utf-8", errors='replace')
' 根'
```

bytes 내의 일반 토큰에서 해당 토큰의 ID로의 매핑은 `tokenizer.get_vocab()`을 통해 검색하실 수 있으며, 일반 토큰을 어휘(vocab)에 추가하는 것을 지원하거나 권장하지 않습니다.

## 특수 토큰, Special tokens
특수 토큰은 공백이나 문서의 끝 따위를 표시하는 등 모델에서 특별한 기능을 표시하는 토큰을 의미합니다.
이론적으로, 특수 토큰들은 입력 텍스트에 존재하지 않아야 하며, 입력 텍스트가 tokenizer에 의해 처리된 후에만 나타나게 됩니다. 특수 토큰 중 하나로 문서의 끝을 표시하는 토큰의 경우 `<|endoftext|>`과 같은 표현 형태(surface forms) 사용하는데, end of text와 같이 실제로 등장할 수 있는 문구의 일부를 특수 토큰에 사용한 이유는 참조의 용이성을 높이기 위함일 뿐으로 만약 참조 용이성을 고려하지 않는다면, 실제 텍스트에 존재하지 않는 임의의 토큰으로 지정해도 무방합니다. `<|endoftext|>`는 현재 Qwen-7B에서 사용되는 특수 토큰으로, Qwen-7B-Chat에서는  `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`가 특수 토큰으로 사용되는데, 이는 해당 모델에게 확정된 의미를 가지고 있기 때문이며, 해당 특수 토큰을 사용해야만 합니다. 다른 목적으로 우리 연구팀은 `<|extra_0|>` 부터 `<|extra_204|>`까지의 추가적인 특수 토큰을 표시하였으며, 이 특수 토큰을 연구팀이 원하는 대로 사용할 수 있습니다.
 `str` 안에서의 특수 토큰의 표현에 대응되는 tokenizer 내의 ID로의 매핑은 tokenizer.special_tokens 메서드를 통해 검색하실 수 있습니다.

`bos`, `eos`, `unk`, `pad`, `mask`, `sep` 등의 고전적인 특수 토큰 개념은 Qwen의 사전 훈련된 모델들(Qwen-7B 및 Qwen-7B-Chat)에 적용되지 않았으나, `pad` 토큰의 경우 이론적으로 모델이 `pad` 토큰을 인지하거나 계산하지 않기때문에 어떤 알려진 토큰이라도 사용해도 상관 없습니다.
하지만 안전을 위해, 우리팀은 tokenizer의 초기화 과정에서 지정된 특수 토큰의 값을 알려진 특수 토큰으로 제한했으며,
미세 조정 시 특수 토큰의 지정이 필요한 다른 프레임워크에서 특수 토큰을 지정하고자 하는 경우, 다음과 같이 지정할 수 있습니다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True, pad_token='<|endoftext|>')
```


> 경고: 사전 훈련된 모델에 대해 `bos`, `eos`, `unk` 등을 설정하는 것은 무의미하며,
> 해당 특수 토큰의 의미를 모델에 지정하는 미세 조정 없이 설정한다면 이상하게 작동할 수 있습니다.
> 특히, 문장의 끝과 여러 문장을 포함할 수 있는 문서의 끝이 여러분의 시나리오에서 동일하다고 확신하지 않는 한, `eos`로 ``로 사용해서는 안 됩니다.

## Injection attack prevention

특수 토큰은 일반 토큰과 다르기 때문에, 특수 토큰이 입력 텍스트에 나타날 경우 어떤 일이 발생할까요?
예를 들어, 다음과 같은 텍스트가 있다고 가정해보겠습니다.

```
print("<|endoftext|>")
```

위 내용은 원래는 다음과 같이 토크나이징되어야 하고,

```
ids:[1350, 9639, 91, 8691, 723, 427, 91, 82598]
tokens: [b'print', b'("<', b'|', b'endo', b'ft', b'ext', b'|', b'>")']
```

아래와 같이 토크나이징되면 안됩니다. (아래의 경우 '<|endoftext|>'가 일반 토큰으로 인지되어야 함에도 불구하고, 특수 토큰으로 인지되어서 151643의 토큰 ID로 매핑된 사례)

```
ids: [1350, 445, 151643, 899]
tokens: [b'print', b'("', '<|endoftext|>', b'")']
```

우리의 기본 설정은 올바른 방식을 사용했습니다. 즉, 특수 토큰의 표현 형태(surface forms)를 일반 텍스트처럼 처리하고, 특수 토큰은 텍스트의 토큰화 이후에 개발자가 직접 처리해야 합니다.
그러나 이것은 커뮤니티 내에서 (비록 안전하지 않을지라도) 관행에 어긋났으므로, 개발자들의 코드를 재사용을 돕기 위한 추가적인 단계를 추가했습니다.

모든 알려진 특수 토큰의 표현 형태를 특수 토큰으로 파싱하도록 기본 동작을 변경하였습니다.
인젝션 방지를 활성화하려면, 토크나이저 호출 시 `allowed_special=set()`을 다음과 같이 전달하세요.


```python
>>> tokenizer('print("<|endoftext|>")', allowed_special=set())
{'input_ids': [1350, 9639, 91, 8691, 723, 427, 91, 82598], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
```

`allowed_special`로 `str`의 집합을 전달함으로써, 보다 세밀하게 동작을 제어할 수 있습니다.

```python
>>> tokenizer('print("<|extra_0|>")<|endoftext|>', allowed_special={'<|endoftext|>'})
{'input_ids': [1350, 9639, 91, 15460, 62, 15, 91, 82598, 151643], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

`disallowed_special`로 `str`의 컬렉션을 전달함으로써, 다음과 같이 입력 텍스트 내에서 특정 특수 토큰의 표현 형태가 발견될 경우 토크나이저가 오류를 발생시키도록 할 수도 있습니다.

```python
>>> tokenizer('print("<|extra_0|>")<|endoftext|>', allowed_special={'<|endoftext|>'}, disallowed_special=('<|extra_0|>', ))
...
ValueError: Encountered text corresponding to disallowed special token '<|extra_0|>'.
If you want this text to be encoded as a special token, pass it to `allowed_special`, e.g. `allowed_special={'<|extra_0|>', ...}`.
If you want this text to be encoded as normal text, disable the check for this token by passing `disallowed_special=(enc.special_tokens_set - {'<|extra_0|>'})`.
To disable this check for all special tokens, pass `disallowed_special=()`.
```

`allowed_special` 및 `disallowed_special`에 대한 자세한 정보는 [`tiktoken` 문서](https://github.com/openai/tiktoken/blob/095924e02c85617df6889698d94515f91666c7ea/tiktoken/core.py#L75)를 참조하십시오.

새로운 기본 설정은 다음과 같이 되어있습니다.

```python
>>> tokenizer('print("<|endoftext|>")', allowed_special="all", disallowed_special=())
{'input_ids': [1350, 445, 151643, 899], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
```

## Vocabulary Expansion

> 경고: 사용자는 다음을 주의 깊게 읽고, 자신이 무엇을 하고 있는지 인지하여야 하며, 이로인해 발생할 수 있는 모든 위험은 사용자가 책임져야 합니다.
> 어휘를 생성하는 방법을 따를 때 각별히 주의해야 합니다.

Qwen 모델의 토크나이저는 BPE에 기반을 두고 있으며, 어휘에 단어를 추가함으로써 직접 어휘를 확장할 수 없습니다.
토크나이징을 위해서는 중간 병합이 필요하며, 이러한 과정을 위한 정보를 얻기 위해 다음 절차를 따르십시요.


1. 각 줄에 토큰과 그 빈도수가 `\t`로 구분되어 있는 평문 텍스트 파일 `qwen_extra_vocab.txt`를 준비합니다.

   다음과 같은 예시가 있을 때,
   ```
   我是一只猫	20
   你是一只猫	10
   他是一只猫	5
   一只	200
   一只猫	100
   夸张的 比喻手法	20
   ```
   각 어휘의 빈도를 BPE를 사용해서 계산해야 합니다.

   

2. `qwen.tiktoken`과 같은 기본 어휘 파일을 준비하고, 새 토큰의 시작 인덱스를 결정합니다.
   
   Qwen 모델의 어휘에는 151,643개의 일반 토큰과 208개의 제어 토큰이 있습니다.
   간단히 새 토큰의 시작 인덱스를 151,851으로 설정할 수 있습니다.
   물론, 많은 비활성 특수 토큰에 덮어쓸 수 있지만, 비활성 토큰에 덮어 쓰고자 할 경우, 토크나이저 코드를 수정해야 합니다.

3. Run the following command:
   ```
   python add_merges.py qwen.tiktoken qwen_extra.tiktoken qwen_extra_vocab.txt
   ```
   `add_merges.py`는 [여기](examples/add_merges.py)에서 찾을 수 있습니다.
   이 스크립트는 제공된 `qwen_extra_vocab.txt`를 바탕으로 새로운 토크나이저의 병합을 학습하여, 새 토큰들과 새 토큰들의 인덱스를 `qwen_extra.tiktoken`에 저장하게 됩니다.
   (경로는 원하는 대로 수정하세요.)

   이것은 순수한 파이썬 구현이므로, 많은 단어를 추가하고자 할 경우 많이 느릴 수 있습니다.

   사전 토큰화(pre-tokenization)때문에 모든 단어를 추가할 수 있는 것은 아니라는 점에 유의하세요.
   
   다음과 같은 단어를 추가하려고 시도할 경우 다음과 같은 경고가 뜨게 됩니다. 
   ```
   WARNING - 夸张的 比喻手法 would be pre-tokenized to ['夸张的', ' 比喻手法'], and thus cannot be added to vocabulary
   WARNING - word 一只 is already a token b'\xe4\xb8\x80\xe5\x8f\xaa', skipping
   INFO - number of existing merges: 151643
   INFO - number of words for expanding: 4
   DEBUG - (b'\xe4\xb8\x80\xe5\x8f\xaa', b'\xe7\x8c\xab') (一只猫) is selected as the next merge with freq 100
   DEBUG - (b'\xe5\x8f\xaa', b'\xe7\x8c\xab') (只猫) is selected as the next merge with freq 35
   DEBUG - (b'\xe6\x98\xaf\xe4\xb8\x80', b'\xe5\x8f\xaa\xe7\x8c\xab') (是一只猫) is selected as the next merge with freq 35
   DEBUG - (b'\xe6\x88\x91', b'\xe6\x98\xaf\xe4\xb8\x80\xe5\x8f\xaa\xe7\x8c\xab') (我是一只猫) is selected as the next merge with freq 20
   DEBUG - (b'\xe4\xbd\xa0', b'\xe6\x98\xaf\xe4\xb8\x80\xe5\x8f\xaa\xe7\x8c\xab') (你是一只猫) is selected as the next merge with freq 10
   DEBUG - (b'\xe4\xbb\x96', b'\xe6\x98\xaf\xe4\xb8\x80\xe5\x8f\xaa\xe7\x8c\xab') (他是一只猫) is selected as the next merge with freq 5
   INFO - number of newly learned merges: 6
   ```

`qwen_extra.tiktoken`은 다음과 같은 라인을 포함하고 있습니다.
```
5LiA5Y+q54yr 151851
5Y+q54yr 151852
5piv5LiA5Y+q54yr 151853
5oiR5piv5LiA5Y+q54yr 151854
5L2g5piv5LiA5Y+q54yr 151855
5LuW5piv5LiA5Y+q54yr 151856
```

다음과 같은 코드로 extra_vocab_file을 사용하실 수 있습니다.
``` python
from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True, extra_vocab_file="qwen_extra.tiktoken")

>>> len(tokenizer)
151857

>>> tokenizer("我是一只猫")
{'input_ids': [151854], 'token_type_ids': [0], 'attention_mask': [1]}
```

참고: `extra_vocab_file` 인수를 사용하려면 최신 토크나이저 코드, 즉 2013년 10월 8일 이후 버전이 필요합니다.
그렇지 않으면, `qwen_extra.tiktoken`의 내용을 수동으로 `qwen.tiktoken`에 추가해야 합니다(경로는 설정에 따라 다를 수 있음).

물론, 새 토큰이 작동하도록 모델을 미세 조정할 필요가 있습니다.


### Caveats

Qwen의 토크나이저는 다른 토크나이저들, 예를 들어 알려지지 않은 문자에 대해 UTF-8 바이트 시퀀스로 대체하는 SentencePiece와 같은 것들과 달리 직접 UTF-8 바이트 시퀀스에 작동할 수 있습니다. 문제는 빈도수가 매우 적은 데이터에 기반하여 BPE가 계산되게 되면, UTF-8 코드포인트 경계가 올바르게 인식되지 않을 수도 있다는 것입니다. 이론적으로는 데이터 양이 제한된 데이터를 사용하여 확장된 어휘로 미세 조정된 모델의 경우 문제가 발생할 수 있습니다.

예를 들어, 문자열 `一只`의 UTF-8 바이트 시퀀스 `b'\xe4\xb8\x80\xe5\x8f\xaa'`에 대해 `b'\x80\xe5'`가 `一`(`b'\xe4\xb8\x80'`)과 `只` (`b'\xe5\x8f\xaa'`)의 UTF-8 코드포인트를 걸쳐 먼저 병합될 수 있습니다. 일반적으로 알려진 단어에 대해서는 잘 작동하지만, 실제로 알려지지 않은 단어일 경우에는 예상치 못한 이상한 병합이 일어날 수 있으며, 이는 사전 훈련된 모델이 토큰을 이해하지 못하는 결과를 초래할 수 있습니다.

안전을 위해, 추가하려는 모든 단어에 대한 UTF-8 코드포인트를 수집하고, 해당 단어의 등장 빈도수의 합계보다 높은 빈도수로 tokenizer에 새로운 토큰을 추가해주는 것이 좋습니다. 그러나 Qwen의 tokenizer에는 대부분의 중국어 단어가 포함되어 있으므로, 중국어의 경우 단어만 추가하는 것만으로도 충분히 안전할 수 있습니다.

탐구심이 많으신 분들은 주어진 예에서 `一只`가 토큰이지만, `只猫`도 새로운 토큰으로 학습된다는 것을 눈치채셨을 수 있습니다. 그 이유는 `是一`도 Qwen에 토큰으로 존재하면서 동시에 `一只`보다 더 높은 병합 우선순위를 가지기 때문에, `是|一|只|猫`의 병합 경로는 `是一|只|猫 -> 是一|只猫 -> 是一只猫`이 되게 됩니다. (UTF-8 바이트 병합 설명은 생략)

이것은 BPE의 원래 특성일 뿐이고, 토큰의 분포에만 기반을 두고 있으며, 어떤 바이트들이 유효한 UTF-8 코드포인트, 문자, 또는 의미 있는 단어를 형성할 수 있는지에 대한 지식이 없기 때문에 발생하는 문제입니다.

이러한 이유 때문에, ASCII 문자만 포함하는 단어라도, 다른 맥락의 문장에서 다르게 하위 토크나이징 될 수 있습니다.

```python
>>> tokenizer.tokenize("Panda")
[b'P', b'anda']

>>> tokenizer.tokenize(" Panda")
[b' Panda']

>>> tokenizer.tokenize("Pandas")
[b'P', b'andas']

>>> tokenizer.tokenize(" Pandas")
[b' Pand', b'as']
```

이러한 문제는 단순히 위와 같이 제한된 분포의 데이터에서 더 자주 발생할 뿐, 만약 방대한 양의 훈련 데이터를 가지고 있다면, 이러한 BPE의 고질적인 특성은 문제가 되지 않을 것입니다. 

훈련 데이터가 충분하고 다양하다면, BPE 알고리즘은 실제 사용 환경에서 자주 등장하는 바이트 조합을 학습하여 효과적인 토큰화를 수행할 수 있습니다. 그러나 훈련 데이터의 양이 매우 적거나 특정 패턴만 지나치게 반복된다면, 위에서 언급한 바와 같이 이상한 병합이 일어날 수 있으며, 이는 모델이 새로운 또는 드문 텍스트를 처리할 때 예상치 못한 결과를 초래할 수 있습니다.

따라서, 새로운 토큰을 추가하거나 어휘를 확장할 때는 훈련 데이터의 다양성과 충분성을 고려하는 것이 중요합니다. 이는 모델이 더 일반화된 표현을 학습하고, 실제 세계의 다양한 언어 사용 사례에 더 잘 적응할 수 있도록 할 것입니다.