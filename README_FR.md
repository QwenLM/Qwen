<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp ｜ &nbsp<a href="README_JA.md">日本語</a>&nbsp ｜ &nbspFrançais
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



Nous ouvrons notre série **Qwen**, qui comprend désormais **Qwen**, les modèles de langue de base, à savoir **Qwen-7B** et **Qwen-14B**, ainsi que **Qwen-Chat**, les modèles de chat, à savoir **Qwen-7B-Chat** et **Qwen-14B-Chat**. Les liens se trouvent dans le tableau ci-dessus. Cliquez dessus et consultez les fiches des modèles. Nous publions également le **[rapport technique](https://arxiv.org/abs/2309.16609)**. Cliquez sur le lien du document et consultez-le !

En bref, nous disposons de modèles linguistiques solides, qui ont été pré-entraîné de manière stable pour 3 000 milliards de tokens de données multilingues avec une large couverture de domaines, de langues (en particulier le chinois et l'anglais), etc. Ils sont capables d'atteindre des performances compétitives sur des ensembles de données de référence. En outre, nous disposons de modèles de chat alignés sur les préférences humaines basées sur SFT et RLHF (pas encore publiés), qui sont capables de chatter, de créer du contenu, d'extraire des informations, de résumer, de traduire, de coder, de résoudre des problèmes mathématiques, etc. et d'utiliser des outils, de jouer le rôle d'agents ou même code interpreter, etc.

Dans la repo, vous pouvez trouver:

* Comment utiliser Qwen, et profiter de l'inférence simple.
* Détails sur les modèles de quantization, y compris GPTQ et la quantization de KV cache.
* Statistiques sur les performances de l'inférence, y compris la vitesse et la mémoire.
* Tutoriels sur le finetuning, y compris le finetuning de paramètres complets, LoRA, et Q-LoRA.
* Instructions de déploiement, avec l'exemple de vLLM et FastChat.
* Instructions sur la création de démos, y compris WebUI, démo CLI, etc.
* Introduction au service API de DashScope, ainsi que les instructions pour construire une API de type OpenAI pour votre modèle.
* Informations sur Qwen pour l'utilisation d'outils, d'agents et code interpreter.
* Statistiques de l'évaluation de la compréhension du contexte long.
* Contrat de licence.
* ...

En outre, si vous rencontrez des problèmes, consultez d'abord la [FAQ](FAQ.md) pour obtenir de l'aide. Vous vous sentez toujours en difficulté ? N'hésitez pas à nous envoyer des questions (de préférence en anglais pour que plus de gens puissent vous comprendre) ! Si vous souhaitez nous aider, envoyez-nous des demandes d'extension sans hésitation ! Nous sommes toujours enthousiastes à propos des relations publiques ! 

Vous voulez discuter avec nous ou prendre un café avec nous ? Bienvenue sur notre Discord ou WeChat !
<br><br>

## Nouvelles et mises à jour

* 2023.10.17 Nous publions le modèle quantifié Int8 **Qwen-7B-Chat-Int8** et **Qwen-14B-Chat-Int8**.
* 2023.9.25 🔥 Nous publions **Qwen-14B** et **Qwen-14B-Chat** sur ModelScope et Hugging Face, ainsi que [qwen.cpp](https://github.com/QwenLM/qwen.cpp) et [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent). Les codes et les poids de **Qwen-7B** et **Qwen-7B-Chat** ont également été mis à jour. **S'IL VOUS PLAÎT, TIREZ LA DERNIÈRE VERSION!**
    - Par rapport à **Qwen-7B** (original), **Qwen-7B** utilise davantage de jetons d'entraînement, passant de 2,2 à 2,4T de jetons, tandis que la longueur du contexte passe de 2048 à 8192. La connaissance du chinois et la capacité de codage de **Qwen-7B** ont été encore améliorées.
* 2023.9.12 Nous prenons désormais en charge le finetuning sur les modèles Qwen-7B, y compris le finetuning de tous les paramètres, LoRA et Q-LoRA.
* 2023.8.21 Nous publions le modèle quantifié Int4 pour Qwen-7B-Chat, **Qwen-7B-Chat-Int4**, qui nécessite de faibles coûts de mémoire mais permet d'améliorer la vitesse d'inférence. En outre, il n'y a pas de dégradation significative des performances lors de l'évaluation de référence.
* 2023.8.3 Nous publions **Qwen-7B** et **Qwen-7B-Chat** sur ModelScope et Hugging Face. Nous fournissons également un mémo technique pour plus de détails sur le modèle, y compris les détails de l'entraînement et les performances du modèle.
<br>

## Performance

Qwen-14B et Qwen-7B (il s'agit de la nouvelle version entraînée avec davantage de tokens et la longueur du contexte est passée de 2048 à 8192) surpassent les modèles de référence de tailles similaires sur une série d'ensembles de données de référence, par exemple MMLU, C-Eval, GSM8K, MATH, HumanEval, MBPP, BBH, etc., qui évaluent les capacités des modèles en matière de compréhension du langage naturel, de résolution de problèmes mathématiques, de codage, etc. Cependant, même Qwen-14B reste nettement inférieur à GPT-3.5, sans parler de GPT-4. Voir les résultats ci-dessous.

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

Pour tous les modèles comparés, nous indiquons les meilleurs scores entre leurs résultats officiels et [OpenCompass] (https://opencompass.org.cn/leaderboard-llm). 

Pour plus de résultats expérimentaux (performances détaillées des modèles sur d'autres ensembles de données de référence) et de détails, veuillez vous référer à notre rapport technique en cliquant [ici](https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf).
<br><br>

## Besoins

* python 3.8 et plus
* pytorch 1.12 et plus, 2.0 et plus sont recommandés
* transformers 4.32 et plus
* CUDA 11.4 et plus sont recommandés (pour les utilisateurs de GPU, les utilisateurs de flash, etc.)
<br>

## Démarrage Rapide

Ci-dessous, nous fournissons des exemples simples pour montrer comment utiliser Qwen-Chat avec 🤖 ModelScope et 🤗 Transformers.

Avant d'exécuter le code, assurez-vous d'avoir configuré l'environnement et installé les paquets requis. Assurez-vous que vous répondez aux exigences ci-dessus, puis installez les bibliothèques dépendantes.

```bash
pip install -r requirements.txt
```

Si votre appareil supporte fp16 ou bf16, nous vous recommandons d'installer [flash-attention](https://github.com/Dao-AILab/flash-attention) (**nous supportons flash-attention 2 maintenant.**) pour une meilleure efficacité et une moindre utilisation de la mémoire. (**flash-attention est optionnel et le projet peut fonctionner normalement sans l'installer**)

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# pip install csrc/rotary
```

Vous pouvez maintenant commencer avec ModelScope ou Transformers.

### 🤗 Transformers

Pour utiliser Qwen-Chat pour l'inférence, il vous suffit de saisir quelques lignes de code, comme indiqué ci-dessous. N'oubliez pas de transmettre les noms de modèles ou les chemins corrects, tels que "Qwen/Qwen-7B-Chat" et "Qwen/Qwen-14B-Chat". Cependant, **veuillez vous assurer que vous utilisez le code le plus récent**.

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

L'exécution du modèle pré-entraîné de Qwen est également simple.

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

En cas de problème de réseau lors de la tentative de téléchargement des poids et des codes du modèle à partir de HuggingFace, une autre approche consiste à récupérer le point de contrôle à partir de ModelScope, puis à le charger à partir du répertoire local, comme indiqué ci-dessous:

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

ModelScope est une plateforme opensource pour Model-as-a-Service (MaaS), qui fournit un service de modèle flexible et rentable aux développeurs d'IA. De même, vous pouvez exécuter les modèles avec ModelScope comme indiqué ci-dessous:

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

### Inférence par lots
Qwen prend en charge l'inférence par lots. Lorsque flash attention est activée, l'utilisation de l'inférence par lots peut entraîner une accélération de 40 %. Le code d'exemple est présenté ci-dessous:
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

Pour déployer nos modèles sur CPU, nous vous conseillons vivement d'utiliser [qwen.cpp](https://github.com/QwenLM/qwen.cpp), qui est une implémentation purement C++ de Qwen et de tiktoken. Consultez le repo pour plus de détails!

Il est simple d'exécuter directement le modèle sur le CPU, ce qui nécessite la spécification de votre appareil:

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
```

Cependant, il est probable que vous souffriez d'une efficacité d'inférence extrêmement faible.

### Plusieurs GPU

Si vous souffrez d'un manque de mémoire GPU et que vous souhaitez exécuter le modèle sur plus d'un GPU, vous pouvez utiliser directement la méthode de chargement par défaut, qui est maintenant supportée par Transformers. La méthode précédente basée sur `utils.py` est obsolète.

Cependant, bien que cette méthode soit simple, l'efficacité du parallélisme natif du pipeline est faible. Nous vous conseillons d'utiliser vLLM avec FastChat et de lire la section relative au déploiement.
<br><br>

## Quantization

### GPTQ

Nous proposons une solution basée sur [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), et publions les modèles quantifiés Int4, qui permettent d'obtenir des effets de modèle presque sans perte mais des performances améliorées en termes de coûts de mémoire et de vitesse d'inférence.

Nous démontrons ici comment utiliser les modèles quantifiés que nous fournissons pour l'inférence. Avant de commencer, assurez-vous que vous répondez aux exigences d'auto-gptq (par exemple, torch 2.0 et plus, transformers 4.32.0 et plus, etc.) et installez les paquets requis:

```bash
pip install auto-gptq optimum
```

Si vous rencontrez des problèmes pour installer `auto-gptq`, nous vous conseillons de consulter le [repo](https://github.com/PanQiWei/AutoGPTQ) officiel pour trouver une roue.

Vous pouvez ensuite charger facilement le modèle quantifié et lancer l'inférence comme d'habitude:

```python
# Model names: "Qwen/Qwen-7B-Chat-Int4", "Qwen/Qwen-14B-Chat-Int4"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "Hi", history=None)
```

Nous illustrons les performances des modèles BF16, Int8 et Int4 sur le benchmark, et nous constatons que le modèle quantifié ne souffre pas d'une dégradation significative des performances. Les résultats sont présentés ci-dessous:

| Quantization         | MMLU | CEval (val) | GSM8K | Humaneval |
|----------------------|:----:|:-----------:|:-----:|:---------:|
| Qwen-7B-Chat (BF16)  | 55.8 |    59.7     | 50.3  |   37.2    |
| Qwen-7B-Chat (Int8)  | 55.4 |    59.4     | 48.3  |   34.8    |
| Qwen-7B-Chat (Int4)  | 55.1 |    59.2     | 49.7  |   29.9    |
| Qwen-14B-Chat (BF16) | 64.6 |    69.8     | 60.1  |   43.9    |
| Qwen-14B-Chat (Int8) | 63.6 |    68.6     | 60.0	 |   48.2    |
| Qwen-14B-Chat (Int4) | 63.3 |    69.0     | 59.8  |   45.7    |

### Quantization du cache KV

Attention Le cache KV peut être quantifié et compressé pour le stockage, afin d'obtenir un débit d'échantillonnage plus élevé. Les paramètres `use_cache_quantization` et `use_cache_kernel` sont fournis pour contrôler le comportement de quantification du cache KV
Lorsque `use_cache_quantization=True` et `use_cache_kernel=True`, la quantization de kv-cache est activée.
La méthode d'utilisation spécifique est la suivante:

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
Attention : Actuellement, la quantization du cache kv et le flash attn ne peuvent pas être activés en même temps.
Si vous activez la quantification du cache kv et use_flash_attn en même temps (`use_flash_attn=True`, `use_cache_quantization=True`, `use_cache_kernel=True`), use_flash_attn est désactivé par défaut (`use_flash_attn=false`).

Nous avons vérifié que l'utilisation du modèle int8-kvcache quantifié ne souffre pas d'une dégradation significative des performances dans l'évaluation en aval. En outre, nous évaluons ses performances en nous concentrant sur l'empreinte mémoire. 
Le profilage s'exécute sur un seul GPU A100-SXM4-80G avec PyTorch 2.0.1 et CUDA 11.4. 
Nous utilisons des modèles BF16, et générons 1024 tokens (seq-length=1024) par défaut, et oom indique qu'il n'y a plus de mémoire.

Lorsque la quantization de kv-cache est activée, nous pouvons utiliser une taille de lot (bs) plus importante.

| USE KVCache |  bs=1  |  bs=4  | bs=16  | bs=32  | bs=64  | bs=100 |
|-------------|:------:|:------:|:------:|:------:|:------:|:------:|
| no          | 16.3GB | 24.1GB | 31.7GB | 48.7GB |  oom   |  oom   |
| yes         | 15.5GB | 17.2GB | 22.3GB | 30.2GB | 48.2GB | 72.4GB |

Lorsque la quantification de kv-cache est activée, le modèle peut économiser plus de mémoire lorsqu'il génère des séquences plus longues (sl, nombre de jetons générés) lors de l'inférence.

| USE KVCache | sl=512 | sl=1024 | sl=2048 | sl=4096 | sl=8192 |
|-------------|:------:|:-------:|:-------:|:-------:|:-------:|
| no          | 15.2GB | 16.3GB  | 17.6GB  | 19.5GB  | 23.2GB  |
| yes         |  15GB  | 15.5GB  | 15.8GB  | 16.6GB  | 17.6GB  |

Le modèle qui active la quantification du kv-cache convertit le format du layer-past de float à int8, tandis que le layer-past quantifié stocke également les paramètres de quantification de la valeur actuelle.
Les étapes spécifiques sont les suivantes :

1. Quantifier clé/valeur
```
    qv,scale,zero_point=quantize_cache_v(v)
```
2. Stocker dans layer_past

Following is the format of quantized layer_past:
```
    layer_past=((q_key,key_scale,key_zero_point),
                (q_value,value_scale,value_zero_point))
```
Format de base de layer_past:
```
    layer_past=(key,value)
```
Si vous souhaitez utiliser l'attention KV qui est quantifiée, vous pouvez utiliser l'opération de déquantification pour convertir la clé/valeur int8 en format float comme suit 
vous pouvez utiliser l'opération de déquantification pour reconvertir la clé/valeur int8 au format float comme suit :
```
    v=dequantize_cache_torch(qv,scale,zero_point)
```
<br>


## Performance de l'inférence

Cette section fournit les statistiques de vitesse et de mémoire des modèles dans différentes précisions. Le profilage de la vitesse et de la mémoire est effectué à l'aide de [ce script](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py).

### Vitesse

Nous avons mesuré la vitesse moyenne d'inférence (jetons/s) pour la génération de 2048 et 8192 jetons avec les modèles dans la précision de BF16, Int8, et Int4 sous la condition d'utiliser l'attention flash v1, v2, ou de ne pas l'utiliser.

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


En détail, le profilage consiste à encoder 2048 jetons et à générer 8192 nouveaux jetons. Le profilage s'exécute sur un seul GPU A100-SXM4-80G avec PyTorch 2.0.1 et CUDA 11.8. La vitesse d'inférence est calculée en moyenne sur les jetons encodés et générés.

Note : La vitesse de génération des modèles Int4/Int8 mentionnés ci-dessus est fournie par la bibliothèque autogptq. La vitesse actuelle du modèle chargé à l'aide de "AutoModelForCausalLM.from_pretrained" sera environ 20% plus lente. Nous avons signalé ce problème à l'équipe HuggingFace et nous le mettrons à jour rapidement si une solution est disponible.

### Utilisation de la mémoire du GPU

Nous avons également établi le profil de l'utilisation maximale de la mémoire du GPU pour l'encodage de 2048 jetons en tant que contexte (et la génération d'un seul jeton) et la génération de 8192 jetons (avec un seul jeton en tant que contexte) sous BF16, Int8 ou Int4 niveau de quantization, respectivement. Les résultats (GB) sont présentés ci-dessous.

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

### Utilisation
Nous fournissons maintenant le script d'entraînement officiel, `finetune.py`, pour que les utilisateurs puissent ajuster le modèle pré-entraîné pour les applications en aval de manière simple. De plus, nous fournissons des scripts shell pour lancer le finetune sans soucis. Ce script prend en charge l'entraînement avec [DeepSpeed](https://github.com/microsoft/DeepSpeed) et [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/). Les scripts que nous fournissons utilisent DeepSpeed (Note : il peut y avoir des conflits avec la dernière version de pydantic) et Peft. Vous pouvez les installer en procédant comme suit:
```bash
pip install peft deepspeed
```

Pour préparer vos données d'entraînement, vous devez rassembler tous les échantillons dans une liste et l'enregistrer dans un fichier json. Chaque échantillon est un dictionnaire composé d'un identifiant et d'une liste de conversation. Voici un exemple simple de liste avec 1 échantillon:
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

Après la préparation des données, vous pouvez utiliser les scripts shell fournis pour lancer le finetuning. N'oubliez pas de spécifier le chemin d'accès au fichier de données, `$DATA`.

Les scripts de finetuning vous permettent d'effectuer les opérations suivantes
- Finetuning de tous les paramètres
- LoRA
- Q-LoRA

Le finetuning de tous les paramètres nécessite la mise à jour de tous les paramètres au cours de l'ensemble du processus de formation. Pour lancer votre formation, exécutez le script suivant:

```bash
# Distributed training. We do not provide single-GPU training script as the insufficient GPU memory will break down the training.
sh finetune/finetune_ds.sh
```

N'oubliez pas de spécifier le nom ou le chemin d'accès au modèle, le chemin d'accès aux données, ainsi que le répertoire de sortie dans les scripts shell. Une autre chose à noter est que nous utilisons DeepSpeed ZeRO 3 dans ce script. Si vous voulez faire des changements, il suffit de supprimer l'argument `--deepspeed` ou de faire des changements dans le fichier json de configuration de DeepSpeed en fonction de vos besoins. De plus, ce script supporte l'entraînement en précision mixte, et donc vous pouvez utiliser `--bf16 True` ou `--fp16 True`. N'oubliez pas d'utiliser DeepSpeed lorsque vous utilisez fp16 en raison de l'entraînement de précision mixte. Empiriquement, nous vous conseillons d'utiliser bf16 pour rendre votre apprentissage cohérent avec notre pré-entraînement et notre alignement si votre machine supporte bf16, et nous l'utilisons donc par défaut.

Pour exécuter LoRA, utilisez un autre script à exécuter comme indiqué ci-dessous. Avant de commencer, assurez-vous que vous avez installé `peft`. Vous devez spécifier les chemins d'accès à votre modèle, à vos données et à vos résultats. Nous vous conseillons d'utiliser des chemins absolus pour votre modèle pré-entraîné. En effet, LoRA ne sauvegarde que l'adaptateur et le chemin absolu dans le fichier json de configuration de l'adaptateur est utilisé pour trouver le modèle pré-entraîné à charger. De plus, ce script supporte à la fois bf16 et fp16.

```bash
# Single GPU training
sh finetune/finetune_lora_single_gpu.sh
# Distributed training
sh finetune/finetune_lora_ds.sh
```

Par rapport au finetuning de tous les paramètres, LoRA ([paper](https://arxiv.org/abs/2106.09685)) ne met à jour que les paramètres des couches d'adaptateurs, tout en gelant les couches originales du grand modèle de langage. Cela permet de réduire considérablement les coûts de mémoire et donc les coûts de calcul.

Notez que si vous utilisez LoRA pour affiner le modèle de langue, par exemple Qwen-7B, au lieu des modèles de chat, par exemple Qwen-7B-Chat, le script change automatiquement les embedding et la couche de sortie en tant que paramètres entraînables. En effet, le modèle de langue n'a aucune connaissance des jetons spéciaux apportés par le format ChatML. Ces couches doivent donc être mises à jour pour que le modèle comprenne et prédise les jetons. En d'autres termes, si votre entraînement apporte des tokens spéciaux dans LoRA, vous devez définir les couches comme des paramètres entraînables en définissant `modules_to_save` à l'intérieur du code. De plus, si ces paramètres sont entraînables, il n'est pas possible d'utiliser ZeRO 3, et c'est pourquoi nous utilisons ZeRO 2 par défaut dans le script. Si vous n'avez pas de nouveaux paramètres entraînables, vous pouvez passer à ZeRO 3 en modifiant le fichier de configuration de DeepSpeed. En outre, nous constatons qu'il existe un écart important entre l'empreinte mémoire de LoRA avec et sans ces paramètres d'entraînement. Par conséquent, si vous avez des problèmes de mémoire, nous vous conseillons d'affiner les modèles de chat de LoRA. Consultez le profil ci-dessous pour plus d'informations.

Si vous souffrez toujours d'un manque de mémoire, vous pouvez envisager Q-LoRA ([paper](https://arxiv.org/abs/2305.14314)), qui utilise le modèle de langage quantifié et d'autres techniques telles que l'attention paginée pour réduire encore les coûts de mémoire.

Note : pour exécuter l'entraînement Q-LoRA sur un seul GPU, vous pouvez avoir besoin d'installer `mpi4py` via `pip` ou `conda`.

Pour lancer Q-LoRA, exécutez directement le script suivant:

```bash
# Single GPU training
sh finetune/finetune_qlora_single_gpu.sh
# Distributed training
sh finetune/finetune_qlora_ds.sh
```

Pour Q-LoRA, nous vous conseillons de charger le modèle quantifié que nous fournissons, par exemple Qwen-7B-Chat-Int4. Vous **NE DEVRIEZ PAS** utiliser les modèles bf16. Contrairement au finetuning de tous les paramètres et à la LoRA, seul le modèle fp16 est pris en charge pour la Q-LoRA. Pour l'entraînement sur un seul GPU, nous devons utiliser DeepSpeed pour l'entraînement en précision mixte en raison de notre observation des erreurs causées par torch amp. En outre, pour Q-LoRA, les problèmes avec les jetons spéciaux dans LoRA existent toujours. Cependant, comme nous ne fournissons que les modèles Int4 pour les modèles de chat, ce qui signifie que le modèle de langage a appris les tokens spéciaux du format ChatML, vous n'avez pas à vous soucier des couches. Notez que les couches du modèle Int4 ne doivent pas être entraînables, et donc si vous introduisez des tokens spéciaux dans votre entraînement, Q-LoRA risque de ne pas fonctionner.

Contrairement au finetuning des paramètres complets, l'entraînement de LoRA et de Q-LoRA n'enregistre que les paramètres de l'adaptateur. Supposons que votre entraînement commence à partir de Qwen-7B, vous pouvez charger le modèle finalisé pour l'inférence comme indiqué ci-dessous:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()
```

Si vous souhaitez fusionner les adaptateurs et enregistrer le modèle affiné en tant que modèle autonome (vous ne pouvez le faire qu'avec LoRA, et vous **NE POUVEZ PAS** fusionner les paramètres de Q-LoRA), vous pouvez exécuter les codes suivants:

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

Note : Pour l'entraînement multi-GPU, vous devez spécifier les hyperparamètres appropriés pour l'entraînement distribué en fonction de votre machine. De plus, nous vous conseillons de spécifier votre longueur maximale de séquence avec l'argument `--model_max_length`, en fonction de votre considération des données, de l'empreinte mémoire, et de la vitesse d'apprentissage.

### Profilage de la mémoire et de la vitesse
Nous profilons la mémoire du GPU et la vitesse d'apprentissage de LoRA (LoRA (emb) se réfère à l'apprentissage de l'embedding et la couche de sortie, tandis que LoRA n'a pas de couche d'intégration et de sortie pouvant être entraînée) et de Q-LoRA dans la configuration de l'apprentissage sur un seul GPU. Dans ce test, nous expérimentons sur un seul GPU A100-SXM4-80G, et nous utilisons CUDA 11.8 et Pytorch 2.0. Flash attention 2 est appliqué. Nous utilisons uniformément une taille de lot de 1 et une accumulation de gradient de 8. Nous profilons la mémoire (GB) et la vitesse (s/iter) des entrées de différentes longueurs, à savoir 256, 512, 1024, 2048, 4096, et 8192. Nous présentons également les statistiques du finetuning de tous les paramètres avec Qwen-7B sur 2 GPU A100. Nous ne présentons que les statistiques de 256, 512 et 1024 jetons en raison de la limitation de la mémoire du GPU. Les statistiques sont listées ci-dessous :

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

## Déploiement

### vLLM 
Pour le déploiement et l'inférence rapide, nous suggérons d'utiliser vLLM avec FastChat. Installez d'abord les paquets:
```bash
pip install vllm
pip install "fschat[model_worker,webui]"
```
Ou vous pouvez les installer à partir des sources par `git clone` et `pip install -e .`. Nous vous conseillons de lire leurs documents si vous rencontrez des problèmes lors de l'installation.

Pour faire fonctionner Qwen avec vLLM et FastChat, vous devez d'abord lancer un contrôleur par:
```bash
python -m fastchat.serve.controller
```

Ensuite, vous pouvez lancer le travailleur de modèle, ce qui signifie charger votre modèle pour l'inférence. Pour l'inférence sur un seul GPU, vous pouvez directement lancer:
```bash
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code
```
Cependant, si vous souhaitez exécuter le modèle sur plusieurs GPU pour une inférence plus rapide ou une mémoire plus importante, vous pouvez utiliser le parallélisme tensoriel pris en charge par vLLM. Supposons que vous exécutiez le modèle sur 4 GPU, la commande est présentée ci-dessous:
```bash
python -m fastchat.serve.vllm_worker --model-path $model_path --trust-remote-code --tensor-parallel-size 4
```

Après avoir lancé votre model worker, vous pouvez lancer une démo web ou une API OpenAI comme vous le souhaitez. Pour la démo web, exécutez la commande suivante:
```bash
python -m fastchat.serve.gradio_web_server
```
Pour l'API OpenAI, consultez d'abord la documentation de notre API OpenAI pour l'installation. Exécutez ensuite la commande:
```bash
python -m fastchat.serve.openai_api_server --host localhost --port 8000
```
<br>

## Démo

### Interface Web

Nous fournissons du code pour que les utilisateurs puissent construire une démo d'interface web (merci à @wysaid). Avant de commencer, assurez-vous d'installer les paquets suivants:

```
pip install -r requirements_web_demo.txt
```

Exécutez ensuite la commande ci-dessous et cliquez sur le lien généré:

```bash
python web_demo.py
```

<p align="center">
    <br>
    <img src="assets/web_demo.gif" width="600" />
    <br>
<p>

### Démo CLI

Nous fournissons un exemple de démonstration CLI dans `cli_demo.py`, qui prend en charge la sortie en continu pour la génération. Les utilisateurs peuvent interagir avec Qwen-7B-Chat en saisissant des invites, et le modèle renvoie les sorties du modèle en mode streaming. Exécutez la commande ci-dessous:

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

Le moyen le plus simple d'utiliser Qwen via les API est le service API DashScope via Alibaba Cloud. Nous présentons une introduction à l'utilisation. De plus, nous fournissons un script pour vous permettre de déployer une API de type OpenAI sur vos propres serveurs.

### DashScope
DashScope est le service API de grands modèles linguistiques fourni par Alibaba Cloud, qui prend désormais en charge Qwen. Notez que les modèles derrière DashScope sont des versions internes temporairement sans détails fournis. Les services comprennent `qwen-turbo` et `qwen-plus`, le premier fonctionnant plus rapidement et le second atteignant de meilleures performances. Pour plus d'informations, consultez la documentation [ici] (https://dashscope.aliyun.com).

Veuillez vous rendre sur le site officiel [lien](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.6c2774fahtfXdn) pour créer un compte DashScope et obtenir la clé API (AK). Nous recommandons de définir l'AK à l'aide d'une variable d'environnement:
```bash
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
```
Installez ensuite les paquets et cliquez sur [ici](https://help.aliyun.com/zh/dashscope/developer-reference/install-dashscope-sdk) pour obtenir la documentation. Si vous utilisez Python, vous pouvez installer DashScope avec pip:
```bash
pip install dashscope
```
Si vous utilisez JAVA SDK, vous pouvez l'installer de cette manière:
```xml
<!-- https://mvnrepository.com/artifact/com.alibaba/dashscope-sdk-java -->
<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>dashscope-sdk-java</artifactId>
    <version>the-latest-version</version>
</dependency>
```
La manière la plus simple d'utiliser DashScope est l'utilisation de messages, qui est similaire à l'API OpenAI. L'exemple est présenté ci-dessous:
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
Pour d'autres utilisations, veuillez consulter le site web officiel pour plus de détails.

### API OpenAI

Nous fournissons des méthodes pour déployer une API locale basée sur l'API OpenAI (merci à @hanpenggit). Avant de commencer, installez les paquets nécessaires:

```bash
pip install fastapi uvicorn openai "pydantic>=2.3.0" sse_starlette
```

Exécutez ensuite la commande pour déployer votre API:

```bash
python openai_api.py
```

Vous pouvez modifier vos arguments, par exemple, `-c` pour le nom ou le chemin du poids, `--cpu-only` pour le déploiement CPU, etc. Si vous rencontrez des problèmes lors du lancement du déploiement de l'API, la mise à jour des paquets vers la dernière version peut probablement les résoudre.

L'utilisation de l'API est simple. Voir l'exemple ci-dessous:

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

**Function calling** est aussi supporté (mais seulement quand `stream=False` pour le moment). Voir [l'exemple d'utilisation](examples/function_call_examples.py) ici.
<br><br>


## Utilisation des outils

Qwen-Chat a été optimisé pour l'utilisation d'outils et les capacités d'appel de fonctions. Les utilisateurs peuvent développer des agents, des applications LangChain, et même augmenter Qwen avec un Code Interpreter.

Nous fournissons une documentation sur la manière d'implémenter les appels d'outils basés sur le principe de ReAct Prompting, veuillez vous référer à [l'exemple ReAct](examples/react_prompt.md). Sur la base de ce principe, nous fournissons un support pour function calling dans [openai_api.py](openai_api.py).

Nous avons testé les capacités d'appel d'outil du modèle sur notre benchmark d'évaluation chinois à source ouverte et nous avons constaté que Qwen-Chat obtient systématiquement de bons résultats:

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

Pour évaluer la capacité de Qwen à utiliser l'interpréteur de code Python pour des tâches telles que la résolution de problèmes mathématiques, la visualisation de données et d'autres tâches générales telles que la manipulation de fichiers et l'exploration du Web, nous avons créé et mis en libre accès un test de référence spécialement conçu pour évaluer ces capacités. Vous pouvez trouver le benchmark sur ce [lien](https://github.com/QwenLM/Qwen-Agent/tree/main/benchmark).

Nous avons observé que Qwen est performant en termes d'exécutabilité du code et de précision des résultats lors de la génération du code:

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

En outre, nous fournissons également des résultats expérimentaux démontrant que notre modèle est capable d'agir en tant qu'agent Hugging Face. Pour plus d'informations, veuillez vous référer à la [documentation de l'exemple](examples/transformers_agent.md). Les performances du modèle sur l'ensemble des données d'évaluation fournies par Hugging Face sont les suivantes:

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

## Compréhension du contexte long

Pour étendre la longueur du contexte et briser le goulot d'étranglement de la longueur de la séquence d'entraînement, nous introduisons plusieurs techniques, y compris l'interpolation consciente de NTK, l'attention de fenêtre, et l'échelle d'attention LogN, pour étendre la longueur du contexte de Qwen-7B/14B de 2k à plus de 8k tokens, et Qwen-7B de 8k à 32k tokens. Nous menons des expériences de modélisation du langage sur l'ensemble de données arXiv avec l'évaluation PPL et nous constatons que Qwen peut atteindre des performances exceptionnelles dans le scénario d'un contexte long. Les résultats sont présentés ci-dessous :

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

Notre tokenizer basé sur tiktoken est différent des autres tokenizers, par exemple le tokenizer sentencepiece. Vous devez faire attention aux tokens spéciaux, en particulier lors de la mise au point. Pour des informations plus détaillées sur le tokenizer et son utilisation dans le cadre du finetuning, veuillez vous référer à la [documentation](tokenization_note.md).
<br><br>

## Reproduction

Pour reproduire les performances du modèle sur des ensembles de données de référence, nous fournissons des scripts permettant de reproduire les résultats. Consultez [eval/EVALUATION.md](eval/EVALUATION.md) pour plus d'informations. Notez que la reproduction peut entraîner de légères différences par rapport à nos résultats.
<br><br>

## FAQ

Si vous rencontrez des problèmes, veuillez vous référer à la [FAQ](FAQ.md) et aux problèmes pour trouver une solution avant de lancer un nouveau problème.
<br><br>

## Citation
Si vous trouvez notre travail utile, n'hésitez pas à nous citer.

```
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```
<br>

## Accord de Licence

Les chercheurs et les développeurs sont libres d'utiliser les codes et les poids des modèles de Qwen et de Qwen-Chat. Nous autorisons également leur utilisation commerciale. Consultez notre licence à [LICENSE](LICENSE) pour plus de détails. Si vous avez des exigences en matière d'utilisation commerciale, veuillez remplir le formulaire ([7B](https://dashscope.console.aliyun.com/openModelApply/qianwen), [14B](https://dashscope.console.aliyun.com/openModelApply/Qwen-14B-Chat)) pour en faire la demande.
<br><br>

## Contactez-nous

Si vous souhaitez laisser un message à notre équipe de recherche ou à notre équipe produit, rejoignez nos groupes Discord ou WeChat! N'hésitez pas non plus à envoyer un courriel à qianwen_opensource@alibabacloud.com.

