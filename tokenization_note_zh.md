# Tokenization

> 注：作为术语的“tokenization”在中文中尚无共识的概念对应，本文档采用英文表达以利说明。

Qwen-7B采用UTF-8字节级别的BPE tokenization方式，并依赖`tiktoken`这一高效的软件包执行分词。
Qwen-7B中有两类token，即源于BPE、`bytes`类型的普通token和特殊指定、`str`类型的特殊token。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)
```

## 普通token

普通token源于BPE，是在UTF-8编码的文本字节序列上学习得到的。
尽管基于字节序列的方式保证了所有文本均可被tokenize且没有未登录token问题，但处理罕见文本时有可能回退到字节级别的编码。
由于从字节序列解码为文本时，`errors`参数设为`replace`，处理不完整的token序列可能会遇到UTF-8解码错误，表象是生成中包含“替换字符”(�)。
这一行为可以通过将`errors`参数设为`ignore`来规避。
一次性修改可以传入tokenizer的`decode`函数，持久性修改可以传入tokenizer的初始化函数，请注意`decode`的配置优先级更高。
`errors`的可选值，请参阅[Python文档](https://docs.python.org/3/library/stdtypes.html#bytes.decode).

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

`bytes`类型的普通token到id的映射可以通过`tokenizer.get_vocab()`获取。
尚不支持也不推荐向tokenizer增加普通token。

## 特殊token

特殊token用以给模型传递特殊信号，如到达文本末尾。
理论上，输入文本中不包含特殊token，它们仅在tokenization后由开发者手动加入。
特殊token的字面表达，如表示文本结束的`<|endoftext|>`，仅便于指代特殊token，不意味着它们在输入文本空间中。
目前，训练中使用的、已经有固定含义的、不应做它用的特殊token，Qwen-7B中有`<|endoftext|>`，Qwen-7B-Chat中有`<|endoftext|>`、`<|im_start|>`以及`<|im_end|>`。
但词表中也留有供扩展的特殊token位，可用`<|extra_0|>`到`<|extra_204|>`来指代。
`str`类型的特殊token字面表达到id的映射，可以通过`tokenizer.special_tokens`获取。

对于提供的模型参数(Qwen-7B和Qwen-7B-Chat)而言，诸如`bos`、`eos`、`unk`、`pad`、`mask`、`sep`等的特殊token的概念并不适用。
特例是`pad`，由于这个token理论上并不参与模型计算，所以可以使用任意token表达这一概念。
但保险起见，目前可在tokenizer初始化时设定的特殊token，仅可使用已知的特殊token字面表达，即`<|endoftext|>`、`<|im_start|>`、`<|im_end|>`和`<|extra_0|>`到`<|extra_204|>`。
对于微调或者其它需要这些token才能运行的框架，可以如下配置

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True, pad_token='<|endoftext|>')
```

> 注意: 对于提供的训练好的模型，设置诸如`bos`、`eos`、`unk`之类的没有意义，即模型不需要这些概念。
> 如果设置了这些token，但没有相应的微调这些token以让模型理解其含义，未知行为可能被触发。
> 特别时，不应混淆`<|endoftext|>`和`eos`的概念，除非应用场景中它们的实际含义是一致的，即句子末尾等价于文本末尾。

**注入攻击防御**

由于特殊token和普通token概念上的差异，如果输入文本中含有特殊token的字面表达该如何处理？
以下面文本为例

```
print("<|endoftext|>")
```

其正确的tokenization为

```
ids:[1350, 9639, 91, 8691, 723, 427, 91, 82598]
tokens: [b'print', b'("<', b'|', b'endo', b'ft', b'ext', b'|', b'>")']
```

不是

```
ids: [1350, 445, 151643, 899]
tokens: [b'print', b'("', '<|endoftext|>', b'")']
```

默认行为曾是正确的，即输入文本中任何字符一律按普通token处理，特殊token应由开发者在tokenization人工处理。
然后，这与社区中的实践似有差异，为开发者复用代码增加了额外适配步骤。

默认行为已被调整为从输入文本中解析特殊token的字面表达。
如需启用注入攻击防御，请传入参数`allowed_special=set()`：

```python
>>> tokenizer('print("<|endoftext|>")', allowed_special=set())
{'input_ids': [1350, 9639, 91, 8691, 723, 427, 91, 82598], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
```

这一行为可以更精细的调控，将`allowed_special`设计为`str`的集合即可：

```python
>>> tokenizer('print("<|extra_0|>")<|endoftext|>', allowed_special={'<|endoftext|>'})
{'input_ids': [1350, 9639, 91, 15460, 62, 15, 91, 82598, 151643], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

如果希望输入中遇到特殊token的字面表达时，获得更直接的提醒，通过配置`disallowed_special`可以让tokenizer直接触发异常：

```python
>>> tokenizer('print("<|extra_0|>")<|endoftext|>', allowed_special={'<|endoftext|>'}, disallowed_special=('<|extra_0|>', ))
...
ValueError: Encountered text corresponding to disallowed special token '<|extra_0|>'.
If you want this text to be encoded as a special token, pass it to `allowed_special`, e.g. `allowed_special={'<|extra_0|>', ...}`.
If you want this text to be encoded as normal text, disable the check for this token by passing `disallowed_special=(enc.special_tokens_set - {'<|extra_0|>'})`.
To disable this check for all special tokens, pass `disallowed_special=()`.
```

更多关于`allowed_special`和`disallowed_special`的信息, 请参阅[`tiktoken`代码](https://github.com/openai/tiktoken/blob/095924e02c85617df6889698d94515f91666c7ea/tiktoken/core.py#L75).

新的默认行为与以下设定等价

```python
>>> tokenizer('print("<|endoftext|>")', allowed_special="all", disallowed_special=())
{'input_ids': [1350, 445, 151643, 899], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
```

