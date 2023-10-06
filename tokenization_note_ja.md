# トークン化

Qwen-7B は `tiktoken` パッケージを使用して、UTF-8 バイトを BPE トークン化します。
Qwen-7B には 2 種類のトークンがあります。BPE の通常のトークン (`bytes` 型) と特殊/制御トークン (`str` 型) です。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)
```

## 通常のトークン

通常のトークンは、UTF-8 エンコーディングでエンコードされたテキストのバイト列から学習した BPE トークンです。
これによってすべてのテキストをトークン化することができ、未知のトークンは存在しませんが、一般的でないテキストをトークン化するときにシングルバイトを使用するようにフォールバックすることがあります。
UTF-8 のデコードエラーに遭遇することがあり、そのエラーのデフォルトは `replace` であるため、不完全な生成では置換文字 (�) が使用されます。
この動作は `errors="ignore"` を `decode` 関数に渡すことで変更することができる。
`errors` のオプションについては、[Python ドキュメント](https://docs.python.org/3/library/stdtypes.html#bytes.decode) を参照してください。

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

通常のトークン (`bytes` 単位) からその ID へのマッピングは `tokenizer.get_vocab()` から取得できます。
通常のトークンを語彙に追加することはサポートしていませんし、推奨もしていません。

## 特別なトークン

特別なトークンは、例えば文書の最後に到達するなど、モデルにとって特別な機能を意味します。
理論的には、これらは入力テキストには存在せず、入力テキストが処理された後にのみ現れます。
例えば、文書の終わりを表す `<|endoftext|>` のような表面的な形は、参照を容易にするためだけのものである。
現在、Qwen-7B では `<|endoftext|>` が、Qwen-7B-Chat では `<|endoftext|>`, `<|im_start|>`, `<|im_end|>` が特殊トークンとして使われています。
他の目的のために、`<|extra_0|>` から `<|extra_204|>` までの特別なトークンを保持しています。
特殊トークンの表面形式 (`str` 内) から ID へのマッピングは `tokenizer.special_tokens` から取得できます。

`bos`、`eos`、`unk`、`pad`、`mask`、`sep` などの概念は学習済みモデル（Qwen-7B と Qwen-7B-Chat）には適用できません。
しかし、`pad` トークンは話が別です。理論的には、モデルがこのトークンを見たり計算したりすることはないので、既知のトークンを使用することができます。
しかし、安全のために、トークナイザーの初期化で指定する特別なトークンの値は、既知の特別なトークンに限定します。
微調整やその他のフレームワークで特別なトークンを必要とする場合は、次のように指定できます

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True, pad_token='<|endoftext|>')
```

> 警告: 私たちが事前に学習したモデルでは、`bos`, `eos`, `unk` などを設定しても意味がありません。
> 特に、`<<endoftext|>` を `eos` のように使ってはいけません。
> 特に `<|endoftext|>` を `eos` として使用することは、文末と文末が同じであると確信できる場合を除き、避けるべきです。

## インジェクション攻撃の防止

特殊トークンは通常のトークンとは異なるため、コントロールトークンの表面形が入力テキストに現れるとどうなるでしょうか？
例えば、次のようなテキストがあるとします

```
print("<|endoftext|>")
```

これは次のようにしてトークン化する必要があります

```
ids:[1350, 9639, 91, 8691, 723, 427, 91, 82598]
tokens: [b'print', b'("<', b'|', b'endo', b'ft', b'ext', b'|', b'>")']
```

こちらではありません

```
ids: [1350, 445, 151643, 899]
tokens: [b'print', b'("', '<|endoftext|>', b'")']
```

つまり、特殊トークンの表面形は通常のテキストと同じように扱い、特殊トークンはテキストのトークン化後に開発者が処理するというものです。
しかし、これはコミュニティにおける（安全ではないとはいえ）慣習に抵触し、開発者が車輪を再利用するための新たなステップを追加することになります。

デフォルトの動作は、すべての既知の特殊トークンの表面形を特殊トークンとして解析するように変更されました。
インジェクション防止を有効にするには、トークナイザーの呼び出しに `allowed_special=set()` を渡します:

```python
>>> tokenizer('print("<|endoftext|>")', allowed_special=set())
{'input_ids': [1350, 9639, 91, 8691, 723, 427, 91, 82598], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
```

`str` のセットを `allowed_special` として渡すことで、きめ細かく動作を制御することができます

```python
>>> tokenizer('print("<|extra_0|>")<|endoftext|>', allowed_special={'<|endoftext|>'})
{'input_ids': [1350, 9639, 91, 15460, 62, 15, 91, 82598, 151643], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

`str` のコレクションを `disallowed_special` として渡すことで、特定の特殊なトークンの表形式が入力テキストで遭遇した場合にトークナイザーがエラーを発生するようにすることもできます

```python
>>> tokenizer('print("<|extra_0|>")<|endoftext|>', allowed_special={'<|endoftext|>'}, disallowed_special=('<|extra_0|>', ))
...
ValueError: Encountered text corresponding to disallowed special token '<|extra_0|>'.
If you want this text to be encoded as a special token, pass it to `allowed_special`, e.g. `allowed_special={'<|extra_0|>', ...}`.
If you want this text to be encoded as normal text, disable the check for this token by passing `disallowed_special=(enc.special_tokens_set - {'<|extra_0|>'})`.
To disable this check for all special tokens, pass `disallowed_special=()`.
```

`allowed_special` と `disallowed_special` の詳細については、[`tiktoken` ドキュメント](https://github.com/openai/tiktoken/blob/095924e02c85617df6889698d94515f91666c7ea/tiktoken/core.py#L75)を参照してください。

新しいデフォルトは以下の通り

```python
>>> tokenizer('print("<|endoftext|>")', allowed_special="all", disallowed_special=())
{'input_ids': [1350, 445, 151643, 899], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
```

