# Fine-tuning Qwen Using Deepspeed


## TL;DR

We provide the official training script `finetune.py` and serveral notebooks that can be leveraged for users to finetune pre-trained models for downstream applications in a simple fashion. The algorithms that we support include full-parameter fine-tuning, LoRA fine-tuning and Q-LoRA fine-tuning. Here is the matrix of our notebooks used in different settings:

| Algorithm | Single GPU | Multiple GPUs|
| --- | --- | --- |
| Full-parameter Fine-tuning | [finetune_fullparameter_single_gpu](finetune_fullparameter_single_gpu.ipynb) | [finetune_fullparameter_multi_gpu](finetune_fullparameter_multi_gpu.ipynb) |
| LoRA Fine-tuning | [finetune_lora_single_gpu](finetune_lora_single_gpu.ipynb) | [finetune_lora_multi_gpu](finetune_lora_multi_gpu.ipynb) |
| Q-LoRA Fine-tuning | [finetune_qlora_single_gpu](finetune_qlora_single_gpu.ipynb) | [finetune_qlora_multi_gpu](finetune_qlora_multi_gpu.ipynb) |

## Requirements

### Environments

The basic requirements for running Qwen models include:

- python 3.8 and above
- pytorch 1.12 and above, 2.0 and above are recommended
- transformers 4.32 and above
- CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)

Our notebooks launch fine-tuning with DeepSpeed and Peft.
(Note: this may have conflicts with the latest version of pydantic and you should use make sure `pydantic<2.0`.)
You can install them by:
```bash
pip install peft deepspeed
```

### Settings and GPU Requirements

We first provide the support matrix for different learning settings. Full-parameter fine-tuning requires updating all parameters in the whole training process.
In comparison with full-parameter fine-tuning, LoRA only updates the parameters of adapter layers but keeps the original large language model layers frozen. This allows much fewer memory costs and thus fewer computation costs. If you still suffer from insufficient memory, you can consider Q-LoRA, which uses the quantized large language model to allow even fewer memory costs. Generally, the GPU consumption rule for tuning Qwen is as follows: full parameter > full parameter (ZeRO2) > full parameter (ZeRO3) > LoRA > LoRA (ZeRO2) > LoRA (ZeRO3) > Q-LoRA > Q-LoRA (ZeRO2).

| Setting | Full-parameter | LoRA | Q-LoRA |
| --- | --- | --- | --- |
| Base | Yes (up to ZeRO3) | Yes (up to ZeRO2) | No |
| Chat | Yes (up to ZeRO3) | Yes (up to ZeRO3) | No |
| Chat-Int4/8 | No | No | Yes |

Here are some useful suggestions for choosing different fine-tuning settings based on GPU memory, espcially for users with GeForce RTX 3090/4090 (24GB) GPUs (or similar), and A100 (80GB) GPUs (or similar). In the experiments, we uniformly use a batch size of 1, gradient accumulation of 16, and max length of 512. Other parameters are set as the same shown in our notebooks. The results are as follows.

| GPU Memory | Number of GPUs |  Qwen-1.8B-Chat | Qwen-7B-Chat | Qwen-14B-Chat | Qwen-72B-Chat |
| --- | --- | --- | --- | --- |  --- |
| 24GB | *1 | Full Parameter | LoRA | Q-LoRA | N/A |
| 24GB | *2 | Full Parameter | LoRA | Q-LoRA | N/A |
| 24GB | *4 | Full Parameter | LoRA | LoRA (w/ ZeRO3) | N/A |
| 80GB | *1 | Full Parameter | LoRA | LoRA | Q-LoRA |
| 80GB | *2 | Full Parameter | Full Parameter (w/ ZeRO3) | LoRA (w/ ZeRO2) | TBD |
| 80GB | *4 | Full Parameter | Full Parameter (w/ ZeRO2) | Full Parameter (w/ ZeRO3) | LoRA (w/ ZeRO3) |

Using other configurations of LoRA/Q-LoRA and ZeRO stages will easily result in failures.


## Data Preparation

To prepare your training data, you need to put all the samples into a list and save it to a json file. Each sample is a dictionary consisting of an id and a list for conversation. Below is a simple example list with 1 sample:
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

You can also use multi-turn conversations as the training set. Here is a simple example:

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
        "value": "你好！我是一名AI助手，我叫通义千问，有需要请告诉我。"
      },
      {
        "from": "user",
        "value": "你都能做什么"
      },
      {
        "from": "assistant",
        "value": "我能做很多事情，包括但不限于回答各种领域的问题、提供实用建议和指导、进行多轮对话交流、文本生成等。"
      }
    ]
  }
]
```


## Single-GPU Training

In the single-GPU training setting, we provide three notebooks:

- [finetune_fullparameter_single_gpu](finetune_fullparameter_single_gpu.ipynb)
- [finetune_lora_single_gpu](finetune_lora_single_gpu.ipynb)
- [finetune_qlora_single_gpu](finetune_qlora_single_gpu.ipynb)

### Full-parameter Fine-tuning

To launch your training, run the following command (with hyper-parameter settings omitted):
```bash
python finetune.py \
    --model_name_or_path $MODEL \
    --data_path  $DATA \
    --output_dir $OUTPUT
```
Remember to specify the correct model name or path, the data path, as well as the output directory.

### LoRA Fine-tuning

Similarly, to run LoRA, use another notebook to run the command as shown below. Before you start, make sure that you have installed `peft`. Also, you need to specify your paths to your model, data, and output. We advise you to use absolute path for your pre-trained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pre-trained model to load. 
```bash
python finetune.py \
    --model_name_or_path $MODEL \
    --data_path  $DATA \
    --output_dir $OUTPUT \
    --use_lora
```
Note that if you use LoRA to fine-tune the base language model, e.g., Qwen-7B, instead of chat models, e.g., Qwen-7B-Chat, the script automatically switches the embedding and output layer as trainable parameters. This is because the base language model has no knowledge of special tokens brought by ChatML format. Thus these layers should be updated for the model to understand and predict the tokens. Or in another word, if your training brings in special tokens in LoRA, you should set the layers to trainable parameters by setting `modules_to_save` inside the code. Check out the following code in the training script `finetune.py`:
```python
is_chat_model = 'chat' in model_args.model_name_or_path.lower()
if training_args.use_lora:
  if lora_args.q_lora or is_chat_model:
    modules_to_save = None
  else:
    modules_to_save = ["wte", "lm_head"]
    lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
    )
    ...
    model = get_peft_model(model, lora_config)
    ...
```
Pay attention that the script relies on the model path to identify the model type, so please keep `chat` in the chat model paths.



### Q-LoRA Fine-tuning

To run single-GPU Q-LoRA training, you may need to install `mpi4py`. Directly run the following script:
```bash
python finetune.py \
    --model_name_or_path $MODEL \
    --data_path  $DATA \
    --output_dir $OUTPUT \
    --use_lora \
    --q_lora \
    --deepspeed "ds_config_zero2.json"
```

For Q-LoRA, we advise you to load our provided quantized model, e.g., Qwen-7B-Chat-Int4. You **SHOULD NOT** use the bf16 models. Different from full-parameter fine-tuning and LoRA, only fp16 is supported for Q-LoRA. For single-GPU training, we have to use DeepSpeed for mixed-precision training due to our observation of errors caused by torch amp. Besides, for Q-LoRA, the troubles with the special tokens in LoRA still exist. However, as we only provide the Int4 models for chat models, which means the language model has learned the special tokens of ChatML format, you have no worry about the layers. Note that the layers of the Int4 model should not be trainable, and thus if you introduce special tokens in your training, Q-LoRA might not work.


In default, our notebooks provide training codes for Qwen-1.8B-Chat.
You can also run the training script to fine-tune other version of the Qwen-series models. We profile the GPU memory usage of all versions based on our notebooks (without changing any hyper-parameter settings) on a single A800 GPU (80GB). The statistics are listed below:

| Training | Qwen-1.8B-Chat | Qwen-7B-Chat | Qwen-14B-Chat | Qwen-72B-Chat |
| --- | --- | --- | --- | --- |
| Full Parameter | 19.6GB | 76.8GB | OOM | OOM |
| LoRA | 7.4GB | 20.3GB | 34.2GB | OOM |
| Q-LoRA | 6.1GB | 12.5GB | 17.8GB | 61.9GB |


### Merging Weights from LoRA and Q-LoRA


#### Inference with Adapters

Different from full-parameter fine-tuning, the training of both LoRA and Q-LoRA only saves the adapter parameters. Suppose your training starts from Qwen-7B, you can load the fine-tuned model for inference as shown below:
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

response, history = model.chat(tokenizer, "你好", history=None)
```

#### Inference with Merged Weights

If you want to merge the adapters and save the fine-tuned model as a standalone model, take LoRA as an example, you can run the following codes:
```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors.
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
```

The `new_model_directory` directory will contain the merged model weights and module files. Please note that `*.cu` and `*.cpp` files may be missing in the saved files. If you wish to use the KV cache functionality, please manually copy them. Besides, the tokenizer files are not saved in the new directory in this step. You can copy the tokenizer files or use the following code:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)
tokenizer.save_pretrained(new_model_directory)
```
Next, the model with merged weights can be loaded by the following code:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(new_model_directory, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    new_model_directory,
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "你好", history=None)
```

Note that you can not merge weights into quantized models. Instead, we can merge the weights based on the original chat model. Take Qwen-7B-Chat-In4 as an example. 
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

# Here, we load the original Qwen-7B-Chat model, instead of the Qwen-7B-Chat-Int4 model.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
# We merge the learned adapter to the Qwen-7B-Chat.
model = PeftModel.from_pretrained(model, path_to_adapter)
merged_model = model.merge_and_unload()
# We save the model to a new path.
merged_model.save_pretrained(path_to_new_model, max_shard_size="2048MB", safe_serialization=True)
```


## Multi-GPU Training

In the multi-GPU training setting, we provide three notebooks:

- [finetune_fullparameter_multi_gpu](finetune_fullparameter_multi_gpu.ipynb)
- [finetune_lora_multi_gpu](finetune_lora_multi_gpu.ipynb)
- [finetune_qlora_multi_gpu](finetune_qlora_multi_gpu.ipynb)

We use `torchrun` to launch the training job on multiple GPUs:

```bash
# for full-parameter fine-tuning
torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 finetune.py \
    --model_name_or_path $MODEL \
    --data_path  $DATA \
    --output_dir $OUTPUT \
    --deepspeed "ds_config_zero2.json"

# for LoRA fine-tuning
torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 finetune.py \
    --model_name_or_path $MODEL \
    --data_path  $DATA \
    --output_dir $OUTPUT \
    --deepspeed "ds_config_zero2.json" \
    --use_lora

# for Q-LoRA fine-tuning
torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 finetune.py \
    --model_name_or_path $MODEL \
    --data_path  $DATA \
    --output_dir $OUTPUT \
    --deepspeed "ds_config_zero2.json" \
    --use_lora \
    --q_lora
```

For multi-GPU training, you also need to specify proper hyperparameters for distributed training based on your machine. Besides, we advise you to specify your maximum sequence length with the argument `--model_max_length`, based on your consideration of data, memory footprint, and training speed.
For the usage of `torchrun` and distrubuted arguments, please refer to [here](https://pytorch.org/docs/stable/elastic/run.html).
Additionally, we find that there is a significant gap between the memory footprint of LoRA with and without these trainable parameters. Therefore, if you have trouble with memory, we advise you to LoRA fine-tune the chat models. Check the profile below for more information. 


### Multi-node Fine-tuning

Our provided scripts also support multi-node fine-tuning. You can refer to the comments in the scripts to correctly set corresponding arguments and launch the script on each node. For more information about multi-node distributed training, please refer to [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

Note: DeepSpeed ZeRO 3 requires much greater inter-node communication rate than ZeRO 2, which will significantly reduce the training speed in the case of multinode finetuning. Therefore, we do not recommend using DeepSpeed ZeRO 3 configurations in multi-node fine-tuning scripts.

### Profiling of Memory and Speed

We profile the GPU memory and training speed of both LoRA (LoRA (emb) refers to training the embedding and output layer, while LoRA has no trainable embedding and output layer) and Q-LoRA in the setup of single-GPU training. In this test, we experiment on a single A100-SXM4-80G GPU, and we use CUDA 11.8 and Pytorch 2.0. Flash attention 2 is applied. We uniformly use a batch size of 1 and gradient accumulation of 8. We profile the memory (GB) and speed (s/iter) of inputs of different lengths, namely 256, 512, 1024, 2048, 4096, and 8192. We also report the statistics of full-parameter fine-tuning with Qwen-7B on 2 A100 GPUs. We only report the statistics of 256, 512, and 1024 tokens due to the limitation of GPU memory. 

For Qwen-7B, we also test the performance of multi-node fine-tuning. We experiment using two servers, each containing two A100-SXM4-80G GPUs, and the rest of configurations are the same as other Qwen-7B experiments. The results of multi-node fine-tuning are marked as LoRA (multinode) in the table.

For Qwen-72B, we experiment in two ways: 1) LoRA fine-tuning + DeepSpeed ZeRO 3 on 4 A100-SXM4-80G GPUs and 2) Q-LoRA (int4) fine-tuning on a single A100-SXM4-80G GPU. Note that OOM occurs on 4 A100-SXM4-80G GPUs both with LoRA (emb) fine-tuning and LoRA fine-tuning without Deepspeed ZeRO 3 (you can pass `--deepspeed ds_config_zero3.json` to `finetune_lora_ds.sh` to enable DeepSpeed ZeRO 3).

The statistics are listed below:

<table>
    <tr>
      <th rowspan="2">Model Size</th><th rowspan="2">Method</th><th rowspan="2">#Nodes</th><th rowspan="2">#GPUs per node</th><th colspan="6" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">256</th><th align="center">512</th><th align="center">1024</th><th align="center">2048</th><th align="center">4096</th><th align="center">8192</th>
    </tr>
    <tr>
        <th rowspan="4">1.8B</th><td>LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">6.7G / 1.0s/it</td><td align="center">7.4G / 1.0s/it</td><td align="center">8.4G / 1.1s/it</td><td align="center">11.0G / 1.7s/it</td><td align="center">16.2G / 3.3s/it</td><td align="center">21.8G / 6.8s/it</td>
    </tr>
    <tr>
        <td>LoRA (emb)</td>
        <td>1</td><td>1</td>
        <td align="center">13.7G / 1.0s/it</td><td align="center">14.0G / 1.0s/it</td><td align="center">14.0G / 1.1s/it</td><td align="center">15.1G / 1.8s/it</td><td align="center">19.7G / 3.4s/it</td><td align="center">27.7G / 7.0s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">5.8G / 1.4s/it</td><td align="center">6.0G / 1.4s/it</td><td align="center">6.6G / 1.4s/it</td><td align="center">7.8G / 2.0s/it</td><td align="center">10.2G / 3.4s/it</td><td align="center">15.8G / 6.5s/it</td>
    </tr>
    <tr>
        <td>Full-parameter</td>
        <td>1</td><td>1</td>
        <td align="center">43.5G / 2.1s/it</td><td align="center">43.5G / 2.2s/it</td><td align="center">43.5G / 2.2s/it</td><td align="center">43.5G / 2.3s/it</td><td align="center">47.1G / 2.8s/it</td><td align="center">48.3G / 5.6s/it</td>
    </tr>
    <tr>
        <th rowspan="5">7B</th>
        <td>LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">20.1G / 1.2s/it</td><td align="center">20.4G / 1.5s/it</td><td align="center">21.5G / 2.8s/it</td><td align="center">23.8G / 5.2s/it</td><td align="center">29.7G / 10.1s/it</td><td align="center">36.6G / 21.3s/it</td>
    </tr>
    <tr>
        <td>LoRA (emb)</td>
        <td>1</td><td>1</td>
        <td align="center">33.7G / 1.4s/it</td><td align="center">34.1G / 1.6s/it</td><td align="center">35.2G / 2.9s/it</td><td align="center">35.1G / 5.3s/it</td><td align="center">39.2G / 10.3s/it</td><td align="center">48.5G / 21.7s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">11.5G / 3.0s/it</td><td align="center">11.5G / 3.0s/it</td><td align="center">12.3G / 3.5s/it</td><td align="center">13.9G / 7.0s/it</td><td align="center">16.9G / 11.6s/it</td><td align="center">23.5G / 22.3s/it</td>
    </tr>
    <tr>
        <td>Full-parameter</td>
<td>1</td><td>2</td>
<td align="center">139.2G / 4.0s/it</td><td align="center">148.0G / 4.0s/it</td><td align="center">162.0G / 4.5s/it</td><td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
        <td>LoRA (multinode)</td>
        <td>2</td><td>2</td>
        <td align="center">74.7G / 2.09s/it</td><td align="center">77.6G / 3.16s/it</td><td align="center">84.9G / 5.17s/it</td><td align="center">95.1G / 9.25s/it</td><td align="center">121.1G / 18.1s/it</td><td align="center">155.5G / 37.4s/it</td>
    </tr>
    <tr>
        <th rowspan="3">14B</th>
        <td>LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">34.6G / 1.6s/it</td><td align="center">35.1G / 2.4s/it</td><td align="center">35.3G / 4.4s/it</td><td align="center">37.4G / 8.4s/it</td><td align="center">42.5G / 17.0s/it</td><td align="center">55.2G / 36.0s/it</td>
    </tr>
    <tr>
        <td>LoRA (emb)</td>
        <td>1</td><td>1</td>
        <td align="center">51.2 / 1.7s/it</td><td align="center">51.1G / 2.6s/it</td><td align="center">51.5G / 4.6s/it</td><td align="center">54.1G / 8.6s/it</td><td align="center">56.8G / 17.2s/it</td><td align="center">67.7G / 36.3s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">18.7G / 5.3s/it</td><td align="center">18.4G / 6.3s/it</td><td align="center">18.9G / 8.2s/it</td><td align="center">19.9G / 11.8s/it</td><td align="center">23.0G / 20.1s/it</td><td align="center">27.9G / 38.3s/it</td>
    </tr>
    <tr>
        <th rowspan="2">72B</th>
        <td>LoRA + Deepspeed Zero3</td>
        <td>1</td><td>4</td>
        <td align="center">215.4G / 17.6s/it</td><td align="center">217.7G / 20.5s/it</td><td align="center">222.6G / 29.4s/it</td><td align="center">228.8G / 45.7s/it</td><td align="center">249.0G / 83.4s/it</td><td align="center">289.2G / 161.5s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td>
        <td>1</td><td>1</td>
        <td align="center">61.4G / 27.4s/it</td><td align="center">61.4G / 31.5s/it</td><td align="center">62.9G / 41.4s/it</td><td align="center">64.1G / 59.5s/it</td><td align="center">68.0G / 97.7s/it</td><td align="center">75.6G / 179.8s/it</td>
    </tr>
</table>
<br>











