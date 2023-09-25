<p align="left">
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp ｜ &nbsp日本語
</p>
<br><br>

<p align="center">
    <img src="assets/logoqwen.jpg" width="400"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/qwen">ModelScope<a>&nbsp&nbsp | &nbsp&nbsp 📑 Paper&nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-7B-Chat-Demo/summary">Demo</a>
<br>
<a href="assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp ｜ &nbsp&nbsp DingTalk (钉钉) &nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/z3GAxXZ9Ce">Discord</a>&nbsp&nbsp
</p>
<br><br>

<p align="left">
        日本語ドキュメントメンテナー: <a href="https://github.com/eltociear">Ikko Eltociear Ashimine</a> & Junyang Lin
</p>
<br>

|    |                                                              Qwen-Chat                                                               |                                                                Qwen-Chat (Int4)                                                                |                                                            Qwen                                                            |
|----|:------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|
| 7B |  <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 <a>  <a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>  |  <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int4/summary">🤖 <a>  <a href="https://huggingface.co/Qwen/Qwen-7B-Chat-Int4">🤗</a>  |  <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 <a>  <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>  |
| 14B | <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary">🤖 <a>  <a href="https://huggingface.co/Qwen/Qwen-14B-Chat">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int4/summary">🤖 <a>  <a href="https://huggingface.co/Qwen/Qwen-14B-Chat-Int4">🤗</a> | <a href="https://modelscope.cn/models/qwen/Qwen-14B/summary">🤖 <a>  <a href="https://huggingface.co/Qwen/Qwen-14B">🤗</a> |



Qwen-7B**と**Qwen-14B**の**Qwen**シリーズと、**Qwen-7B-Chat**と**Qwen-14B-Chat**の**Qwen-Chat**シリーズをオープンソース化しました。上の表にリンクがあります。クリックしてモデルカードをご確認ください。また、テクニカルレポートも公開しました。論文リンクをクリックしてご覧ください！

簡単に説明すると、私たちは、ドメインや言語（中国語と英語を中心に）などを幅広くカバーする最大3兆トークンの多言語データに対して安定的に事前学習された強力なベース言語モデルを持っています。これらのモデルは、ベンチマークデータセットにおいて競争力のあるパフォーマンスを達成することができます。さらに、SFTとRLHFに基づく人間の嗜好に沿ったチャットモデル（まだリリースされていません）があり、チャット、コンテンツ作成、情報抽出、要約、翻訳、コーディング、数学の問題を解くなどが可能で、ツールを使ったり、エージェントとして遊んだり、コードインタプリタとして遊んだりすることもできます。

このレポでは、それを把握することができる：

* Qwenのクイックスタート。
* 量子化モデルの詳細（使用量、メモリ、推論速度など）。比較のために、BF16モデルの統計も提供します。
* フルパラメーターチューニング、LoRA、Q-LoRAを含む、微調整に関するチュートリアル。
* WebUI、CLIデモなど、デモの構築に関する説明。
* ツール使用、エージェント、コードインタプリタの Qwen の詳細。
* ロングコンテクスト理解評価の統計
* ライセンス契約
* ...

また、困ったことがあれば、まずは[FAQ](FAQ.md)を参照してほしい。まだ悩んでいますか？遠慮なく私たちに問題を送ってください（より多くの人が理解できるように、英語で）！私たちを助けたいなら、遠慮なくプルリクエストを送ってください！

私たちとチャットしたいですか？私たちのDiscordまたはWeChatグループへようこそ！
<br><br>

## ニュースとアップデート

* 2023.9.25 🔥 Qwen-14BとQwen-14B-ChatをModelScopeとHugging Faceでリリースしました。同時に、Qwen-7B と Qwen-7B-Chat も更新しました。Qwen-7B（オリジナル）と比較して、Qwen-7Bはより多くの学習トークンを使用し、2.2Tトークンから2.4Tトークンに増加し、コンテキスト長は2048から8192に拡張された。Qwen-7Bの中国語知識とコーディング能力はさらに向上しています。最新のコードとチェックポイントをお使いください！
* 2023.9.12 Qwen-7Bモデルにおいて、フルパラメーター・ファインチューニング、LoRA、Q-LoRAを含むファインチューニングをサポートしました。
* 2023.8.21 Qwen-7B-Chat 用 Int4 量子化モデル **Qwen-7B-Chat-Int4** をリリースしました。また、ベンチマーク評価においても大きな性能低下は見られませんでした。
* 2023.8.3 ModelScope と Hugging Face 上で **Qwen-7B** と **Qwen-7B-Chat** をリリースしました。また、トレーニングの詳細やモデルの性能など、モデルの詳細については技術メモを提供しています。
<br>

## 性能

Qwen-14BとQwen-7B（これは、より多くのトークンで学習され、コンテキストの長さが2048から8192に拡張された新バージョン）は、自然言語理解、数学的問題解決、コーディングなどに関するモデルの能力を評価する一連のベンチマークデータセット、例えばMMLU、C-Eval、GSM8K、MATH、HumanEval、MBPP、BBHなどにおいて、同様のモデルサイズのベースラインモデルを上回る。しかし、Qwen-14BでもGPT-4はおろかGPT-3.5にも大きく遅れをとっています。以下の結果をご覧ください。

<p align="left">
    <img src="assets/radar_14b.jpg" width="600"/>
<p>
<br>

| Model              |   MMLU   |  C-Eval  |  GSM8K   |   MATH   | HumanEval |   MBPP    |   BBH    |  CMMLU   |
|:-------------------|:--------:|:--------:|:--------:|:--------:|:---------:|:---------:|:--------:|:--------:|
|                    |  5-shot  |  5-shot  |  8-shot  |  4-shot  |  0-shot   |  3-shot   |  3-shot  |  5-shot  |
| LLaMA2-7B          |   46.8   |   32.5   |   16.7   |   3.3    |   12.8    |   20.8    |   38.2   |   31.8   |
| LLaMA2-13B         |   55.0   |   41.4   |   29.6   |   5.0    |   18.9    |   30.3    |   45.6   |   38.4   |
| LLaMA2-34B         |   62.6   |    -     |   42.2   |   6.2    |   22.6    |   33.0    |   44.1   |    -     |
| ChatGLM2-6B        |   47.9   |   51.7   |   32.4   |   6.5    |     -     |     -     |   33.7   |    -     |
| InternLM-7B        |   51.0   |   53.4   |   31.2   |   6.3    |   10.4    |   14.0    |   37.0   |   51.8   |
| InternLM-20B       |   62.1   |   58.8   |   52.6   |   7.9    |   25.6    |   35.6    |   52.5   |   59.0   |
| Baichuan2-7B       |   54.7   |   56.3   |   24.6   |   5.6    |   18.3    |   24.2    |   41.6   |   57.1   |
| Baichuan2-13B      |   59.5   |   59.0   |   52.8   |   10.1   |   17.1    |   30.2    |   49.0   |   62.0   |
| Qwen-7B (original) |   56.7   |   59.6   |   51.6   |   10.4    |   24.4    |   31.2    |   40.6   |   58.8   |
| **Qwen-7B**        |   58.2   |   63.5   |   51.7   |   11.6   |   29.9    |   31.6    |   45.0   |   62.2   |
| **Qwen-14B**       | **66.3** | **72.1** | **61.3** | **24.8** | **32.3**  | **40.8**  | **53.4** | **71.0** |


比較されたすべてのモデルについて、公式に報告された結果と[OpenCompass](https://opencompass.org.cn/leaderboard-llm) の間の最高スコアを報告します。

より詳細な実験結果（より多くのベンチマークデータセットでの詳細なモデル性能）や詳細については、[こちら](TODO)をクリックして技術メモを参照してください。
<br><br>

## 必要条件

* python 3.8 以上
* pytorch 1.12 以上、2.0 以上を推奨
* CUDA 11.4 以上を推奨（GPU ユーザー、フラッシュアテンションユーザー向けなど）
<br>

## クイックスタート

以下では、Qwen-Chat と 🤖 ModelScope と 🤗 Transformers の簡単な使用例を示します。

コードを実行する前に、環境のセットアップと必要なパッケージのインストールが済んでいることを確認してください。上記の要件を満たしていることを確認してから、依存するライブラリをインストールしてください。

```bash
pip install -r requirements.txt
```

お使いのデバイスが fp16 または bf16 をサポートしている場合、[flash-attention](https://github.com/Dao-AILab/flash-attention) をインストールすることで、より高い効率とメモリ使用量を抑えることができます。(**flash-attention はオプションであり、インストールしなくてもプロジェクトは正常に実行できます**)

```bash
git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 以下はオプションです。インストールに時間がかかる場合があります。
# pip install csrc/layer_norm
# pip install csrc/rotary
```

これで ModelScope か Transformers で始めることができます。

#### 🤗 Transformers

Qwen-Chat を推論に使用するには、以下のように数行のコードを入力するだけです。**最新のコードを使用していることを確認してください。**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Model names："Qwen/Qwen-7B-Chat"、"Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# bf16 を使用
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# fp16 を使用
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# CPU のみ使用
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# オートモードを使用すると、デバイスに応じて自動的に精度が選択されます。
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()

# 生成のためのハイパーパラメータを指定
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# 第一回対話ターン
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 第二回対話ターン
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# 第三轮对话 第三回対話ターン
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》
```

Qwen の学習済みベースモデルの実行も簡単です。

<details>
  <summary>Qwen の実行</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Model names："Qwen/Qwen-7B"、"Qwen/Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
# bf16 を使用
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, bf16=True).eval()
# fp16 を使用
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, fp16=True).eval()
# CPU のみ使用
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="cpu", trust_remote_code=True).eval()
# オートモードを使用すると、デバイスに応じて自動的に精度が選択されます。
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True).eval()

# 生成のためのハイパーパラメータを指定
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# 蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是亚的斯亚贝巴（Addis Ababa）...
```

</details>

#### 🤖 ModelScope

ModelScope は、MaaS（Model-as-a-Service） のためのオープンソースプラットフォームであり、AI 開発者に柔軟で費用対効果の高いモデルサービスを提供します。同様に、以下のように ModelScope でモデルを実行することができます:

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

# Model names："Qwen/Qwen-7B-Chat"、"Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", revision='v1.0.5', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", revision='v1.0.5', device_map="auto", trust_remote_code=True, fp16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", revision='v1.0.5', trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

response, history = model.chat(tokenizer, "你好", history=None)
print(response)
response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=history) 
print(response)
response, history = model.chat(tokenizer, "它有什么好玩的景点", history=history)
print(response)
```
<br>

## 量子化

### 使用方法

**注: [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) に基づく新しい解決策を提供し、Qwen-Chat 用の Int4 量子化モデル[ここをクリック](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4)をリリースしました。このモデルは、従来の解決策と比較して、ほぼ無損失のモデル効果を達成しつつ、メモリコストと推論速度の両方で性能が向上しています。**

ここでは、量子化されたモデルを推論に使用する方法を説明する。始める前に、auto-gptqの要件を満たしていることを確認し（例：torch 2.0以上、transformers 4.32.0以上など）、必要なパッケージをインストールしてください：

```bash
pip install auto-gptq optimum
```

auto-gptq`のインストールに問題がある場合は、公式の[repo](https://github.com/PanQiWei/AutoGPTQ)をチェックして、ホイールを見つけることをお勧めする。

そうすれば、量子化されたモデルを簡単にロードすることができ、いつもと同じように推論を実行することができる：

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "Hi", history=None)
```

### 性能

ベンチマークにおける BF16 モデルと Int4 モデルの性能について説明します。その結果は以下に示します：

| Quantization         | MMLU | CEval (val) | GSM8K | Humaneval |
|----------------------|:----:|:-----------:|:-----:|:---------:|
| Qwen-7B-Chat (BF16)  | 53.9 |    54.2     | 41.1  |   24.4    |
| Qwen-7B-Chat (Int4)  | 52.6 |    52.9     | 38.1  |   23.8    |
| Qwen-14B-Chat (BF16) | 64.6 |    69.8     | 61.0  |   43.9    |
| Qwen-14B-Chat (Int4) | 63.3 |    69.0     | 59.8  |   45.7    |

### 推論スピード

BF16 の精度と Int4 の量子化レベルの下で、それぞれ 2048 個と 8192 個のトークンを生成する平均推論速度(tokens/s)を測定しました。

| Quantization         | Speed (2048 tokens) | Speed (8192 tokens) |
|----------------------|:-------------------:|:-------------------:|
| Qwen-7B-Chat (BF16)  |        30.34        |        29.32        |
| Qwen-7B-Chat (Int4)  |        43.56        |        33.92        |
| Qwen-14B-Chat (BF16) |        30.70        |        21.73        |
| Qwen-14B-Chat (Int4) |        37.11        |        26.11        |

詳細には、プロファイリングの設定は、1 コンテクストトークンで 8192 個の新しいトークンを生成しています。プロファイリングは、PyTorch 2.0.1 と CUDA 11.4 を搭載したシングル A100-SXM4-80G GPU で実行されました。推論速度は生成された 8192 個のトークンの平均値となります。

### GPU メモリ使用量

また、BF16またはInt4の量子化レベルで、それぞれ2048トークンをコンテキストとしてエンコードした場合（および単一のトークンを生成した場合）と、8192トークンを生成した場合（単一のトークンをコンテキストとして生成した場合）のGPUメモリ使用量のピーク値をプロファイリングしました。その結果を以下に示します。

| Quantization         | Peak Usage for Encoding 2048 Tokens | Peak Usage for Generating 8192 Tokens |
|----------------------|:-----------------------------------:|:-------------------------------------:|
| Qwen-7B-Chat (BF16)  |               17.66GB               |                22.58GB                |
| Qwen-7B-Chat (Int4)  |               8.21GB                |                13.62GB                |
| Qwen-14B-Chat (BF16) |               30.15GB                 |                38.94GB                  |
| Qwen-14B-Chat (Int4) |               13.00GB                 |                21.79GB                  |

上記のスピードとメモリーのプロファイリングは、[このスクリプト](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py)を使用しています。
<br><br>

## ファインチューニング

現在、公式のトレーニングスクリプト `finetune.py` を提供しています。さらに、finetune.pyのシェルスクリプトを提供し、finetune.pyを実行することで、finetune.pyを起動することができる。さらに、安心してファインチューニングを開始するためのシェルスクリプトも提供しています。このスクリプトは、[DeepSpeed](https://github.com/microsoft/DeepSpeed) および [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/) を使用したトレーニングをサポートします。弊社が提供するシェル・スクリプトは DeepSpeed と Peft を使用するため、事前に DeepSpeed と Peft をインストールすることをお勧めします：
```bash
pip install -r requirements_finetune.txt
```

学習データを準備するには、すべてのサンプルをリストにまとめ、jsonファイルに保存する必要があります。各サンプルはidと会話リストで構成される辞書です。以下は1つのサンプルを含む単純なリストの例です：

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好",
      },
      {
        "from": "assistant",
        "value": "我是一个语言模型，我叫通义千问。"
      }
    ]
  }
]
```

データ準備の後、提供されているシェルスクリプトを使って微調整を実行することができる。データファイルのパス `$DATA` を忘れずに指定してください。

ファインチューニングのスクリプトを使用することで、以下のことが可能になる：
- フルパラメーター・ファインチューニング
- LoRA
- Q-LoRA

フルパラメータパラメータのファインチューニングを行うには、トレーニングプロセス全体ですべてのパラメータを更新する必要があります。トレーニングを開始するには、以下のスクリプトを実行します：

```bash
# 分散トレーニング。GPUメモリが不足するとトレーニングが破綻するため、シングルGPUのトレーニングスクリプトは提供していません。
sh finetune/finetune_ds.sh
```

シェルスクリプトでは、正しいモデル名またはパス、データパス、出力ディレクトリを指定することを忘れないでください。このスクリプトでは DeepSpeed ZeRO 3 を使用しています。変更したい場合は、引数 `--deepspeed` を削除するか、要件に基づいて DeepSpeed 設定 json ファイルを変更してください。さらに、このスクリプトは混合精度のトレーニングに対応しており、`--bf16 True` または `--fp16 True` を使用することができます。経験的に、あなたのマシンがbf16をサポートしている場合、私たちのプリトレーニングとアライメントを整合させるためにbf16を使用することをお勧めします。

同様に、LoRAを実行するには、以下のように別のスクリプトを使って実行する。始める前に、`peft`がインストールされていることを確認してください。また、モデル、データ、出力へのパスを指定する必要があります。学習済みモデルには絶対パスを使用することをお勧めします。なぜなら、LoRAはアダプタのみを保存し、アダプタ設定jsonファイルの絶対パスは、ロードする事前学習済みモデルを見つけるために使用されるからです。また、このスクリプトはbf16とfp16の両方をサポートしている。

```bash
# シングルGPUトレーニング
sh finetune/finetune_lora_single_gpu.sh
# 分散トレーニング
sh finetune/finetune_lora_ds.sh
```

LoRA ([論文](https://arxiv.org/abs/2106.09685)) は、フルパラメーターによるファインチューニングと比較して、adapterのパラメーターを更新するだけで、元の大きな言語モデル層は凍結されたままである。そのため、メモリコストが大幅に削減でき、計算コストも削減できる。しかし、それでもメモリ不足に悩む場合は、Q-LoRA（[論文](https://arxiv.org/abs/2305.14314)）を検討することができます。これは、量子化されたラージ言語モデルと、ページド・アテンションなどの他のテクニックを使用し、さらに少ないメモリコストで実行することができます。Q-LoRAを実行するには、以下のスクリプトを直接実行してください：

```bash
# シングルGPUトレーニング
sh finetune/finetune_qlora_single_gpu.sh
# 分散トレーニング
sh finetune/finetune_qlora_ds.sh
```

Q-LoRAについては、弊社が提供する量子化モデル、例えばQwen-7B-Chat-Int4をロードすることをお勧めします。ただし、フルパラメータ・ファインチューニングやLoRAとは異なり、Q-LoRAではfp16のみがサポートされる。

LoRAとQ-LoRAの学習は、フルパラメータによるファインチューニングとは異なり、アダプターパラメータのみを保存する。仮にQwen-7Bから学習を開始したとすると、以下のようにファインチューニングされたモデルを読み込んで推論を行うことができる：

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()
```

シェルスクリプトは`torchrun`を使用してシングルGPUまたはマルチGPUトレーニングを実行します。そのため、分散トレーニングのための適切なハイパーパラメータをマシンに応じて指定する必要があります。
<br><br>

## デモ

### ウェブ UI

ウェブ UI デモを構築するためのコードを提供します（@wysaid に感謝）。これを始める前に、以下のパッケージがインストールされていることを確認してください:

```bash
pip install -r requirements_web_demo.txt
```

そして、以下のコマンドを実行し、生成されたリンクをクリックします:

```bash
python web_demo.py
```

<p align="center">
    <br>
    <img src="assets/web_demo.gif" width="600" />
    <br>
<p>

### CLI デモ

`cli_demo.py` に CLI のデモ例を用意しています。ユーザはプロンプトを入力することで Qwen-7B-Chat と対話することができ、モデルはストリーミングモードでモデルの出力を返します。以下のコマンドを実行する:

```
python cli_demo.py
```

<p align="center">
    <br>
    <img src="assets/cli_demo.gif" width="600" />
    <br>
<p>
<br>

## API

OpenAI API をベースにローカルAPIをデプロイする方法を提供する（@hanpenggit に感謝）。始める前に、必要なパッケージをインストールしてください:

```bash
pip install fastapi uvicorn openai pydantic sse_starlette
```

それから、API をデプロイするコマンドを実行します:

```bash
python openai_api.py
```

チェックポイント名やパスには `-c`、CPU デプロイメントには `--cpu-only` など、引数を変更できます。API デプロイメントを起動する際に問題が発生した場合は、パッケージを最新バージョンに更新することで解決できる可能性があります。

API の使い方も簡単です。以下の例をご覧ください:

```python
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

# ストリーミングレスポンスを有効化するリクエストを作成する
for chunk in openai.ChatCompletion.create(
    model="Qwen",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
    # ストリーミング出力形式でのストップワードの指定はまだサポートされておらず、開発中です。
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)

# ストリーミングレスポンスを有効化しないリクエストを作成する
response = openai.ChatCompletion.create(
    model="Qwen",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=False,
    stop=[] # 例えば、stop=["Observation:"] (ReAct プロンプトの場合)。
)
print(response.choices[0].message.content)
```

<p align="center">
    <br>
    <img src="assets/openai_api.gif" width="600" />
    <br>
<p>
<br>

## デプロイ

CPU 上でモデルを実行するのは簡単であり、以下のようにデバイスを指定する必要があります:

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
```

メモリ不足に悩まされ、複数の GPU にモデルをデプロイしたい場合は、`utils.py` で提供されているスクリプトを使うことができます:

```python
from utils import load_model_on_gpus
model = load_model_on_gpus('Qwen/Qwen-7B-Chat', num_gpus=2)
```

7B チャットモデルの推論を 2GPU で実行できます。
<br>

## ツールの使用

Qwen-7B-Chat は、API、データベース、モデルなど、ツールの利用に特化して最適化されており、ユーザは独自の Qwen-7B ベースの LangChain、エージェント、コードインタプリタを構築することができます。ツール利用能力を評価するための評価[ベンチマーク](eval/EVALUATION.md)では、Qwen-7B は安定した性能に達しています。

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
    <tr>Qwen-7B-Chat v1.1
        <td>Qwen-7B-Chat v1.1</td><td align="center">98%</td><td align="center">0.91</td><td align="center">7.3%</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">98%</td><td align="center">0.93</td><td align="center">2.4%</td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="4" align="center">Using Code Interpreter - Executable Rate of Generated Code (%)</th>
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
    <tr>Qwen-7B-Chat v1.1
        <td>Qwen-7B-Chat v1.1</td>
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
        <th colspan="4" align="center">Using Code Interpreter - Accuracy of Code Execution Results (%)</th>
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
    <tr>Qwen-7B-Chat v1.1
        <td>Qwen-7B-Chat v1.1</td>
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


ReAct プロンプトの書き方や使い方については、[ReAct の例](examples/react_prompt.md)を参照してください。ツールを使用することで、モデルがよりよいタスクを実行できるようになります。

<p align="center">
    <br>
    <img src="assets/code_interpreter_showcase_001.jpg" />
    <br>
<p>

さらに、エージェントとしての能力を示す実験結果を提供する。詳細は [Hugging Face Agent](examples/transformers_agent.md) を参照して下さい。Hugging Face が提供するランモードベンチマークでの性能は以下の通りです:

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
    <tr>Qwen-7B-Chat v1.1
        <td>Qwen-7B-Chat v1.1</td><td align="center">87.0</td><td align="center">87.0</td><td align="center">71.5</td>
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
    <tr>Qwen-7B-Chat v1.1
        <td>Qwen-7B-Chat v1.1</td><td align="center">94.7</td><td align="center">94.7</td><td align="center">85.1</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">95.5</td>
    </tr>
</table>

<br>

## 長い文脈の理解

コンテキストの長さを拡張し、訓練シーケンスの長さのボトルネックを解消するために、NTK を考慮した補間、ウィンドウアテンション、LogN アテンションスケーリングなどの技術を導入し、コンテキストの長さを 8K トークン以上に拡張する。arXiv データセットを用いて PPL 評価による言語モデリング実験を行い、Qwen-7B が長いコンテキストのシナリオにおいて卓越した性能を達成できることを見出した。以下に結果を示します:

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
        <td>Qwen-7B v1.1</td><td align="center"><b>4.23</b></td><td align="center"><b>3.81</b></td><td align="center"><b>3.52</b></td><td align="center"><b>3.31</b></td><td align="center">7.27</td><td align="center">181.49</td>
    </tr>
    <tr>
        <td>+ dynamic_ntk</td><td align="center"><b>4.23</b></td><td align="center"><b>3.81</b></td><td align="center"><b>3.52</b></td><td align="center"><b>3.31</b></td><td align="center"><b>3.23</b></td><td align="center">3.33</td>
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

## トークナイザー

tiktoken に基づくトークナイザーは、他のトークナイザー、例えばセンテンスピーストークナイザーとは異なります。特にファインチューニングの際には、特殊なトークンに注意を払う必要があります。トークナイザに関する詳細な情報や、ファインチューニングにおける使用方法については、[ドキュメント](tokenization_note_ja.md)を参照してください。
<br><br>

## 再現

ベンチマークデータセットでのモデル性能の再現のために、結果を再現するスクリプトを提供しています。詳しくは [eval/EVALUATION.md](eval/EVALUATION.md) を確認してください。なお、再現の結果、我々の報告結果と若干異なる場合があります。
<br><br>

## FAQ

問題が発生した場合は、まずは [FAQ](FAQ_ja.md) や issue を参照し、新しい issue を立ち上げる前に解決策を探してください。
<br><br>

## ライセンス契約

Qwen と Qwen-Chat のコードとモデルウェイトは、研究者や開発者が自由に使用することができます。また、商用利用も可能です。詳しくは [LICENSE](LICENSE) をご覧ください。商用利用を希望される方は、[リクエストフォーム](https://dashscope.console.aliyun.com/openModelApply/qianwen)に必要事項をご記入の上、お申し込みください。
<br><br>

## お問い合わせ

研究チームまたは製品チームへのメッセージは、qianwen_opensource@alibabacloud.com までお気軽にお送りください。

