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

## Vocabulary Expansion

> WARNING: Read carefully, be aware of what you are doing, and use at your own risk. 
> There are certain caveats regarding how your vocabulary is produced.

The tokenizer of Qwen models are based on BPE and you cannot directly expand the vocabulary by adding words to the vocabulary. 
The intermediate merges are needed for tokenization.
Please follow the steps to obtain such information.

1. Prepare a plain text file `qwen_extra_vocab.txt`, where each line contains a token and its frequency separated by `\t`. 

   An example is given below:
   ```
   我是一只猫	20
   你是一只猫	10
   他是一只猫	5
   一只	200
   一只猫	100
   夸张的 比喻手法	20
   ```
   The frequencies are needed to compute the BPE.

   

2. Prepare the base vocabulary file, e.g., `qwen.tiktoken`, and determine the start index for new tokens.
   
   There are 151,643 regular tokens and 208 control tokens in the vocabulary for Qwen models. 
   For simplicity, the start index can be set as 151,851, which is the default value. 
   You can, of course, override the many inactive control tokens, but you will need to modify the tokenizer code. 

3. Run the following command:
   ```
   python add_merges.py qwen.tiktoken qwen_extra.tiktoken qwen_extra_vocab.txt
   ```
   `add_merges.py` can be found [here](examples/add_merges.py).
   It will learn the new merges based on the provided `qwen_extra_vocab.txt`. 
   The new tokens and their indices will be stored in `qwen_extra.tiktoken`. 
   Modify the paths as you wish.

   It is a pure Python implementation, so please expect it to be slow if you are adding a lot of words.

   Please note that not all words can be added due to pre-tokenization. 
   You will get warnings if you try to add such word:
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

The `qwen_extra.tiktoken` will contain the following lines:
```
5LiA5Y+q54yr 151851
5Y+q54yr 151852
5piv5LiA5Y+q54yr 151853
5oiR5piv5LiA5Y+q54yr 151854
5L2g5piv5LiA5Y+q54yr 151855
5LuW5piv5LiA5Y+q54yr 151856
```

You may use the file as follows in your code:
``` python
from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True, extra_vocab_file="qwen_extra.tiktoken")

>>> len(tokenizer)
151857

>>> tokenizer("我是一只猫")
{'input_ids': [151854], 'token_type_ids': [0], 'attention_mask': [1]}
```
Note: You need the latest tokenizer code, i.e., after 2023-10-08, to use the `extra_vocab_file` argument.
Otherwise, you need to manually append `qwen.tiktoken` (of which path varies with your configuration) with the content from `qwen_extra.tiktoken`.

Certainly, you will need to finetune the model for the new tokens to work.


### Caveats


The tokenizer of Qwen operates directly on UTF-8 byte sequences, unlike others, e.g., SentencePiece that operates on Unicode codepoints/characters and falls back to UTF-8 byte sequences for the unknown (IIRC). 
The thing is if the frequencies are computed on limited data, the Unicode codepoint boundary may not be correctly recognized.
In theory, it could be a problem for fine-tuned models using the expanded vocabulary with limited data.

For example, it could happen that `b'\x80\xe5'` might be merged first for the UTF-8 byte sequence `b'\xe4\xb8\x80\xe5\x8f\xaa'` of the string `一只`, across the Unicode codepoint of `一` (`b'\xe4\xb8\x80'`) and `只` (`b'\xe5\x8f\xaa'`).
Normally, this would work just fine for known tokens, but for actually unknown words, unusual merges may happen, which may not be well understood for the pre-trained model.

Our advice is that to be safe, you should gather the Unicode codepoints from all the words you need to add, and also add them to the file with frequencies higher than the sum of the frequencies of the corresponding words.
But since Qwen has most of the Chinese words, it could be okay to just add the Chinese words alone.

For curious minds, you will also notice that in the given example, `一只` is a token and `只猫` is also learned as a new token. 
The reason is that `是一` is also a token in Qwen and has higher merging priority than `一只`, such that the merging path for `是|一|只|猫` is `是一|只|猫 -> 是一|只猫 -> 是一只猫` (omitting the UTF-8 byte merges).

This is the characteristic for plain BPE: it is based solely on distribution, meaning it does not have knowledge of which bytes can form a valid Unicode codepoint, character, or meaningful word.

The byproduct is that text may be sub-tokenized differently in different contexts, even for words containing only ASCII characters.
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
This simply suggests that those combinations occur more frequently in the data.
If you have vast amount of training data, it should not be a problem.