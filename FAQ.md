# FAQ

## Installation & Environment

#### Failure in installing flash attention

Flash attention is an option for accelerating training and inference. Only NVIDIA GPUs of Turing, Ampere, Ada, and Hopper architecture, e.g., H100, A100, RTX 3090, T4, RTX 2080, can support flash attention. **You can use our models without installing it.**

#### Which version of transformers should I use?

4.32.0 is preferred.

#### I downloaded the codes and checkpoints but I can't load the model locally. What should I do?

Please check if you have updated the code to the latest, and correctly downloaded all the sharded checkpoint files.

#### `qwen.tiktoken` is not found. What is it?

This is the merge file of the tokenizer. You have to download it. Note that if you just git clone the repo without [git-lfs](https://git-lfs.com), you cannot download this file.

#### transformers_stream_generator/tiktoken/accelerate not found

Run the command `pip install -r requirements.txt`. You can find the file at [https://github.com/QwenLM/Qwen-7B/blob/main/requirements.txt](https://github.com/QwenLM/Qwen/blob/main/requirements.txt).
<br><br>



## Demo & Inference

#### Is there any demo? CLI demo and Web UI demo?

Yes, see `web_demo.py` for web demo and `cli_demo.py` for CLI demo. See README for more information.


#### Can I use CPU only?

Yes, run `python  cli_demo.py --cpu-only` will load the model and inference on CPU only.

#### Can Qwen support streaming?

Yes. See the function `chat_stream` in `modeling_qwen.py`.

#### Gibberish in result when using chat_stream().

This is because tokens represent bytes and a single token may be a meaningless string. We have updated the default setting of our tokenizer to avoid such decoding results. Please update the code to the latest version.

#### It seems that the generation is not related to the instruction...

Please check if you are loading Qwen-Chat instead of Qwen. Qwen is the base model without alignment, which behaves differently from the SFT/Chat model.

#### Is quantization supported?

Yes, the quantization is supported by AutoGPTQ. 


#### Slow when processing long sequences

Updating the code to the latest version can help.

#### Unsatisfactory performance in processing long sequences

Please ensure that NTK is applied. `use_dynamc_ntk` and `use_logn_attn` in `config.json` should be set to `true` (`true` by default).
<br><br>



## Finetuning

#### Can Qwen support SFT or even RLHF?

Yes, we now support SFT, including full-parameter finetuning, LoRA, and Q-LoRA. Also you can check other projects like [FastChat](**[https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat)), [Firefly]([https://github.com/yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)), [**LLaMA Efficient Tuning**]([https://github.com/hiyouga/LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)), etc.

However, temporarily we do not support RLHF. We will provide the code in the near future.
<br><br>



## Tokenizer

#### bos_id/eos_id/pad_id not found

In our training, we only use `<|endoftext|>` as the separator and padding token. You can set bos_id, eos_id, and pad_id to tokenizer.eod_id. Learn more about our tokenizer from our documents about the tokenizer.



## Docker

#### Download official docker image is very slow

When downloading our official docker image, you may have a slow download speed due to some network issues. You can refer to [Alibaba Cloud Container Image Service](https://help.aliyun.com/zh/acr/user-guide/accelerate-the-pulls-of-docker-official-images) to accelerate the download of official images.
