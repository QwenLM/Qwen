import argparse
import json
from typing import Dict
import logging

import torch
import transformers
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    data = []
    # input_ids, targets = [], []
    for i, source in enumerate(sources):
        source = source["conversations"]
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id = torch.tensor(input_id[:max_len], dtype=torch.int)
        target = torch.tensor(target[:max_len], dtype=torch.int)
        data.append(dict(input_ids=input_id, attention_mask=input_id.ne(tokenizer.pad_token_id)))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Quantization using AutoGPTQ")
    parser.add_argument("--model_name_or_path", type=str, help="model path")
    parser.add_argument("--data_path", type=str, help="calibration data path")
    parser.add_argument("--out_path", type=str, help="output path of the quantized model")
    parser.add_argument("--max_len", type=int, default=8192, help="max length of calibration data")
    parser.add_argument("--bits", type=int, default=4, help="the bits of quantized model. 4 indicates int4 models.")
    parser.add_argument("--group-size", type=int, default=128, help="the group size of quantized model")
    args = parser.parse_args()
    
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=0.01,
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        static_groups=False,
        sym=True,
        true_sequential=True,
        model_name_or_path=None,
        model_file_base_name="model"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    data = preprocess(json.load(open(args.data_path)), tokenizer, args.max_len)

    model = AutoGPTQForCausalLM.from_pretrained(args.model_name_or_path, quantize_config, device_map="auto", trust_remote_code=True)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    model.quantize(data, cache_examples_on_gpu=False)

    model.save_quantized(args.out_path, use_safetensors=True)
    tokenizer.save_pretrained(args.out_path)
