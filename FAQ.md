# FAQ

## Installation & Environment

#### Failure in installing flash attention

Flash attention is an option for accelerating training and inference. Only NVIDIA GPUs of Turing, Ampere, Ada, and Hopper architecture, e.g., H100, A100, RTX 3090, T4, RTX 2080, can support flash attention. You can use our models without installing it.

#### Which version of transformers should I use?

4.31.0 is preferred.

#### I downloaded the codes and checkpoints but I can't load the model locally. What should I do?

Please check if you have updated the code to the latest, and correctly downloaded all the sharded checkpoint files.

#### `qwen.tiktoken` is not found. What is it?

This is the merge file of the tokenizer. You have to download it. Note that if you just git clone the repo without [git-lfs](https://git-lfs.com), you cannot download this file.

#### transformers_stream_generator/tiktoken/accelerate not found

Run the command `pip install -r requirements.txt`. You can find the file at [https://github.com/QwenLM/Qwen-7B/blob/main/requirements.txt](https://github.com/QwenLM/Qwen-7B/blob/main/requirements.txt).
<br><br>



## Demo & Inference

#### Is there any demo? CLI demo and Web UI demo?

Yes, see `web_demo.py` for web demo and `cli_demo.py` for CLI demo. See README for more information.



#### Can I use CPU only?

Yes, run `python  cli_demo.py --cpu_only` will load the model and inference on CPU only.

#### Can Qwen support streaming?

Yes. See the function `chat_stream` in `modeling_qwen.py`.

#### Gibberish in result when using chat_stream().

This is because tokens represent bytes and a single token may be a meaningless string. We have updated the default setting of our tokenizer to avoid such decoding results. Please update the code to the latest version.

#### It seems that the generation is not related to the instruction...

Please check if you are loading Qwen-7B-Chat instead of Qwen-7B. Qwen-7B is the base model without alignment, which behaves differently from the SFT/Chat model.

#### Is quantization supported?

Yes, the quantization is supported by `bitsandbytes`. We are working on an improved version and will release the quantized model checkpoints.

#### Errors in running quantized models: `importlib.metadata.PackageNotFoundError: No package metadata was found for bitsandbytes`

For Linux users，running `pip install bitsandbytes` directly can solve the problem. For Windows users, you can run `python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui`·

#### Slow when processing long sequences

We solved this problem. Updating the code to the latest version can help.

#### Unsatisfactory performance in processing long sequences

Please ensure that NTK is applied. `use_dynamc_ntk` and `use_logn_attn` in `config.json` should be set to `true` (`true` by default).
<br><br>



## Finetuning

#### Can Qwen support SFT or even RLHF?

We do not provide finetuning or RLHF codes for now. However, some projects have supported finetuning, see [FastChat](**[https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat)), [Firefly]([https://github.com/yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)), [**LLaMA Efficient Tuning**]([https://github.com/hiyouga/LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)), etc. We will soon update the relevant codes.
<br><br>



## Tokenizer

#### bos_id/eos_id/pad_id not found

In our training, we only use `<|endoftext|>` as the separator and padding token. You can set bos_id, eos_id, and pad_id to tokenizer.eod_id. Learn more about our tokenizer from our documents about the tokenizer.

