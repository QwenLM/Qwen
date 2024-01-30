# Quantization

## GPTQ

We provide a solution based on [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), and release the Int4 and Int8 quantized models, which achieve nearly lossless model effects but improved performance on both memory costs and inference speed.

Here we demonstrate how to use our provided quantized models for inference. Before you start, make sure you meet the requirements of auto-gptq (e.g., torch 2.0 and above, transformers 4.32.0 and above, etc.) and install the required packages:

```bash
pip install auto-gptq optimum
```

If you meet problems installing `auto-gptq`, we advise you to check out the official [repo](https://github.com/PanQiWei/AutoGPTQ) to find a wheel.

> Note: The pre-compiled `auto-gptq` packages strongly depend on the version of `torch` and its CUDA version. Moreover, due to recent update, 
> you may also encounter unsupported version errors from `transformers`, `optimum`, or `peft`.
> We recommend using the latest versions meeting the following requirements:
> - torch==2.1 auto-gptq>=0.5.1 transformers>=4.35.0 optimum>=1.14.0 peft>=0.6.1
> - torch>=2.0,<2.1 auto-gptq<0.5.0 transformers<4.35.0 optimum<1.14.0 peft>=0.5.0,<0.6.0

Then you can load the quantized model easily and run inference as same as usual:

```python
# Model names: "Qwen/Qwen-7B-Chat-Int4", "Qwen/Qwen-14B-Chat-Int4"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "Hi", history=None)
```

We illustrate the model performance of both BF16, Int8 and Int4 models on the benchmark, and we find that the quantized model does not suffer from significant performance degradation. Results are shown below:

| Quantization         | MMLU | CEval (val) | GSM8K | Humaneval |
|----------------------|:----:|:-----------:|:-----:|:---------:|
| Qwen-1.8B-Chat (BF16)| 43.3 |    55.6     | 33.7  |   26.2    |
| Qwen-1.8B-Chat (Int8)| 43.1 |    55.8     | 33.0  |   27.4    |
| Qwen-1.8B-Chat (Int4)| 42.9 |    52.8     | 31.2  |   25.0    |
| Qwen-7B-Chat (BF16)  | 55.8 |    59.7     | 50.3  |   37.2    |
| Qwen-7B-Chat (Int8)  | 55.4 |    59.4     | 48.3  |   34.8    |
| Qwen-7B-Chat (Int4)  | 55.1 |    59.2     | 49.7  |   29.9    |
| Qwen-14B-Chat (BF16) | 64.6 |    69.8     | 60.1  |   43.9    |
| Qwen-14B-Chat (Int8) | 63.6 |    68.6     | 60.0  |   48.2    |
| Qwen-14B-Chat (Int4) | 63.3 |    69.0     | 59.8  |   45.7    |
| Qwen-72B-Chat (BF16) | 74.4 |    80.1     | 76.4  |   64.6    |
| Qwen-72B-Chat (Int8) | 73.5 |    80.1     | 73.5  |   62.2    |
| Qwen-72B-Chat (Int4) | 73.4 |    80.1     | 75.3  |   61.6    |

## Quantization of KV cache

> NOTE: Please be aware that due to the internal mechanism of Hugging Face, the support files for this functionality 
> (i.e., `cache_autogptq_cuda_256.cpp` and `cache_autogptq_cuda_kernel_256.cu`) may be missing. Please manually download
> them from the Hugging Face Hub and place them into the same folder as the other module files.

The attention KV cache can be quantized and compressed for storage, to get a higher sample throughput. The arguments `use_cache_quantization` and `use_cache_kernel` in `config.json` are provided to enable KV cache quantization. The specific use method is as follows:
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
Attention: Currently, KV cache quantization and flash attention cannot be used at the same time.
If you enable KV cache quantization and flash attention at the same time (`use_flash_attn=True`, `use_cache_quantization=True`, `use_cache_kernel=True`), `use_flash_attn` is disabled by default (`use_flash_attn=false`).

We have verified that the use of the quantized Int8-KV-Cache model does not suffer from significant performance degradation in downstream evaluation. In the following, we focus on profiling its memory footprint in different conditions. 
The profiling runs on a single A100-SXM4-80G GPU with PyTorch 2.0.1 and CUDA 11.4. 
We use BF16 models to generate 1024 tokens by default, and "OOM" indicates out-of-memory error.

With KV cache quantization, the model can infer with a larger batch size (bs).

| USE KV Cache |  bs=1  |  bs=4  | bs=16  | bs=32  | bs=64  | bs=100 |
|--------------|:------:|:------:|:------:|:------:|:------:|:------:|
| No           | 16.3GB | 24.1GB | 31.7GB | 48.7GB |  OOM   |  OOM   |
| Yes          | 15.5GB | 17.2GB | 22.3GB | 30.2GB | 48.2GB | 72.4GB |

With KV cache quantization the model can save more memory when generating longer sequence (`sl`, sequence length, referring to the number of tokens generated) at the stage of inference.

| USE KV Cache | sl=512 | sl=1024 | sl=2048 | sl=4096 | sl=8192 |
|--------------|:------:|:-------:|:-------:|:-------:|:-------:|
| No           | 15.2GB | 16.3GB  | 17.6GB  | 19.5GB  | 23.2GB  |
| Yes          |  15GB  | 15.5GB  | 15.8GB  | 16.6GB  | 17.6GB  |

The model with KV cache quantization will convert the format of `layer_past` from float to int8, and meanwhile the quantized `layer-past` will also store the quantization parameters.

Specific steps are as follows:

1. Quantize key/value
```
    qv,scale,zero_point=quantize_cache_v(v)
```
2. Store into layer_past

The following is the format of quantized `layer_past`:
```
    layer_past=((q_key,key_scale,key_zero_point),
                (q_value,value_scale,value_zero_point))
```

The original format of `layer_past` is shown below:
```
    layer_past=(key,value)
```

If you want to use the attention KV which is quantized, you can use the dequantization operation to convert the Int8 key/value back to the float format as follows:
```
    v=dequantize_cache_torch(qv,scale,zero_point)
```
<br>