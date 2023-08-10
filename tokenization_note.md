# Tokenization

Qwen-7B uses BPE tokenization on UTF-8 bytes using the `tiktoken` package.
There are two types of tokens in Qwen-7B, i.e., the regular tokens (of type `bytes`) in BPE and the special/control tokens (of type `str`).

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)
```

## Regular tokens

The regular tokens are BPE tokens learned from byte sequences of texts encoded using the UTF-8 encoding.
While this allows tokenization of all texts and no unknown token exists, it may fall back to using single bytes when tokenizing uncommon texts.
You may encounter UTF-8 decoding errors and as the errors are default to `replace`, thus the replacement character (�) in incomplete generation.
You can change this behavior by passing `errors="ignore"` to the `decode` function for once or to the `from_pretrained` function forever.
For more options of `errors`, please refer to [the Python documentation](https://docs.python.org/3/library/stdtypes.html#bytes.decode).

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

The mapping from regular tokens (in `bytes`) to its ID can be retrieved from `tokenizer.get_vocab()`.
We do not support or recommended adding regular tokens to the vocabulary.

## Special tokens

The special tokens signify special functions to the model, e.g., reaching the end of a document.
In theory, they do not exist in the input texts and only appear after the input texts are processed.
Their surface forms, e.g., `<|endoftext|>` for the end of a document, are only meant for ease of reference.
Currently, used special tokens are `<|endoftext|>` in Qwen-7B, and `<|endoftext|>`, `<|im_start|>`, and `<|im_end|>` in Qwen-7B-Chat, which means they have determined meanings to the corresponding model, and should not be used otherwise.
For other purposes, we keep extra special tokens from `<|extra_0|>` to `<|extra_204|>`, and you can use them as you wish.
The mapping from surface forms of the special tokens (in `str`) to its ID can be retrieved from `tokenizer.special_tokens`.

The concepts of `bos`, `eos`, `unk`, `pad`, `mask`, `sep` and such are not appliable to our pretrained models (Qwen-7B and Qwen-7B-Chat).
The `pad` token, however, is a different story, as in theory, the model never sees or computes this token, so you may use any known token.
But to be safe, we limit the value of special tokens specified in the initialization of the tokenizer to the known special tokens.
You may specify special tokens in fine-tuning or in any other frameworks that necessitate them like this

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True, pad_token='<|endoftext|>')
```

> WARNING: For our pretrained models, setting `bos`, `eos`, `unk`, and such makes no sense.
> Unknown behavior may be introduced if you set them without fine-tuning that designates their meanings to the model.
> Especially, you should not use `<|endoftext|>` as `eos`, unless you are sure that the end of a sentence and the end of a document, which may contain many sentences, are the same in your scenario.

## Injection attack prevention

As special tokens are different from regular tokens, what will happen if the surface forms of a control token appear in the input texts?
For example, note that a piece of text like this

```
print("<|endoftext|>")
```

should be tokenized as

```
ids:[1350, 9639, 91, 8691, 723, 427, 91, 82598]
tokens: [b'print', b'("<', b'|', b'endo', b'ft', b'ext', b'|', b'>")']
```

not

```
ids: [1350, 445, 151643, 899]
tokens: [b'print', b'("', '<|endoftext|>', b'")']
```

Our default used to be the correct one, that is, treating the surface forms of special tokens just like regular texts, and special tokens should be taken cared of by developers after tokenization of the texts.
However, this conflicts with (albeit unsafe) practice in the community, and adds another step for developers to reuse their wheels.

The default behavior has been changed to parse the surface forms of all the known special tokens as special tokens.
To enable injection prevention, pass `allowed_special=set()` to the calls of the tokenizer:

```python
>>> tokenizer('print("<|endoftext|>")', allowed_special=set())
{'input_ids': [1350, 9639, 91, 8691, 723, 427, 91, 82598], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
```

You can control the behavior in a fine-grained manner by passing a set of `str` as `allowed_special`

```python
>>> tokenizer('print("<|extra_0|>")<|endoftext|>', allowed_special={'<|endoftext|>'})
{'input_ids': [1350, 9639, 91, 15460, 62, 15, 91, 82598, 151643], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

You can also make the tokenizer raise errors if the surface forms of certain special tokens are encountered in the input texts by passing a collection of `str` as `disallowed_special`

```python
>>> tokenizer('print("<|extra_0|>")<|endoftext|>', allowed_special={'<|endoftext|>'}, disallowed_special=('<|extra_0|>', ))
...
ValueError: Encountered text corresponding to disallowed special token '<|extra_0|>'.
If you want this text to be encoded as a special token, pass it to `allowed_special`, e.g. `allowed_special={'<|extra_0|>', ...}`.
If you want this text to be encoded as normal text, disable the check for this token by passing `disallowed_special=(enc.special_tokens_set - {'<|extra_0|>'})`.
To disable this check for all special tokens, pass `disallowed_special=()`.
```

For more information on `allowed_special` and `disallowed_special`, please refer to [the `tiktoken` documentation](https://github.com/openai/tiktoken/blob/095924e02c85617df6889698d94515f91666c7ea/tiktoken/core.py#L75).

The new default is the same as

```python
>>> tokenizer('print("<|endoftext|>")', allowed_special="all", disallowed_special=())
{'input_ids': [1350, 445, 151643, 899], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
```

