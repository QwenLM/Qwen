# FAQ

## インストールと環境

#### Flash attention 導入の失敗例

Flash attention は、トレーニングと推論を加速するオプションです。H100、A100、RTX 3090、T4、RTX 2080 などの Turing、Ampere、Ada、および Hopper アーキテクチャの NVIDIA GPU だけが、flash attention をサポートできます。それをインストールせずに私たちのモデルを使用することができます。

#### transformers のバージョンは？

4.32.0 が望ましいです。

#### コードとチェックポイントをダウンロードしましたが、モデルをローカルにロードできません。どうすればよいでしょうか？

コードを最新のものに更新し、すべてのシャードされたチェックポイントファイルを正しくダウンロードしたかどうか確認してください。

#### `qwen.tiktoken` が見つかりません。これは何ですか？

これはトークナイザーのマージファイルです。ダウンロードする必要があります。[git-lfs](https://git-lfs.com) を使わずにリポジトリを git clone しただけでは、このファイルをダウンロードできないことに注意してください。

#### transformers_stream_generator/tiktoken/accelerate が見つかりません。

コマンド `pip install -r requirements.txt` を実行してください。このファイルは [https://github.com/QwenLM/Qwen/blob/main/requirements.txt](https://github.com/QwenLM/Qwen/blob/main/requirements.txt) にあります。
<br><br>



## デモと推論

#### デモはありますか？CLI と Web UI のデモはありますか？

はい、Web デモは `web_demo.py` を、CLI デモは `cli_demo.py` を参照してください。詳しくは README を参照してください。



#### CPU のみを使うことはできますか？

はい、`python cli_demo.py --cpu-only` を実行すると、CPU のみでモデルと推論をロードします。

#### Qwen はストリーミングに対応していますか？

`modeling_qwen.py` の `chat_stream` 関数を参照してください。

#### chat_stream() を使用すると、結果に文字化けが発生します。

これは、トークンがバイトを表し、単一のトークンが無意味な文字列である可能性があるためです。このようなデコード結果を避けるため、トークナイザのデフォルト設定を更新しました。コードを最新版に更新してください。

#### インストラクションとは関係ないようですが...

Qwen ではなく Qwen-Chat を読み込んでいないか確認してください。Qwen はアライメントなしのベースモデルで、SFT/Chat モデルとは挙動が異なります。

#### 量子化はサポートされていますか？

はい、量子化は AutoGPTQ でサポートされています。


#### 長いシーケンスの処理に時間がかかる

コードを最新版に更新することで解決します。

#### 長いシーケンスの処理で不満足なパフォーマンス

NTK が適用されていることを確認してください。`config.json` の `use_dynamc_ntk` と `use_logn_attn` を `true` に設定する必要があります（デフォルトでは `true`）。
<br><br>



## ファインチューニング

#### Qwen は SFT、あるいは RLHF に対応できますか？

SFTのコードは提供します。[FastChat](**[https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat))、[Firefly]([https://github.com/yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly))、[**LLaMA Efficient Tuning**]([https://github.com/hiyouga/LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning))など、いくつかのプロジェクトではファインチューニングをサポートしています。近日中に関連コードを更新する予定です。
<br><br>



## トークナイザー

#### bos_id/eos_id/pad_id が見つかりません。

私たちのトレーニングでは、セパレータとパディングトークンとして `<|endoftext|>` のみを使用しています。bos_id、eos_id、pad_id は tokenizer.eod_id に設定できます。私たちのトークナイザーについて詳しくは、トークナイザーについてのドキュメントをご覧ください。

