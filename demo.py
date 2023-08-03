# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed


def demo_qwen_pretrain(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    print("load tokenizer")
    max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="cuda:0",
        max_memory=max_memory,
        trust_remote_code=True,
    ).eval()
    inputs = tokenizer(
        "蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是",
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    pred = model.generate(inputs=inputs["input_ids"])
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))


def demo_qwen_chat(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    print("load tokenizer")
    max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="cuda:0",
        max_memory=max_memory,
        trust_remote_code=True,
    ).eval()
    queries = [
        "请问把大象关冰箱总共要几步？",
        "1+3=?",
        "请将下面这句话翻译为英文：在哪里跌倒就在哪里趴着",
    ]
    history = None
    for turn_idx, query in enumerate(queries, start=1):
        response, history = model.chat(
            tokenizer,
            query,
            history=history,
        )
        print(f"===== Turn {turn_idx} ====")
        print("Query:", query, end="\n")
        print("Response:", response, end="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint-path", type=str, help="Checkpoint path")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")

    args = parser.parse_args()
    set_seed(args.seed)

    if 'chat' in args.checkpoint_path.lower():
        demo_qwen_chat(args)
    else:
        demo_qwen_pretrain(args)
