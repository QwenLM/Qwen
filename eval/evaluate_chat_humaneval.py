import re
import textwrap
import argparse
from pathlib import Path
import tqdm
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

"""
Get the HumanEval.jsonl file from [here](https://github.com/openai/human-eval/tree/master/data)

python eval/evaluate_chat_humaneval.py -f HumanEval.jsonl -o HumanEval_res.jsonl
git clone https://github.com/openai/human-eval
pip install -e human-eval
evaluate_functional_correctness HumanEval_res.jsonl
"""

DEVICE = "cuda:0"


def extract_code(text, entry_point):
    # 正则表达式匹配代码块
    code_block_pattern = re.compile(
        rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
    )
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    # if no code block is found, assume the LM is simply filling the code
    return textwrap.indent(text, " " * 4)


def generate_sample(model, tokenizer, question, entry_point):
    response, _ = model.chat(
        tokenizer,
        question,
        history=None,
    )
    print(question)
    print(response)
    answer = extract_code(response, entry_point)
    return answer, response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=Path,
        help="Checkpoint path",
        default="Qwen/Qwen-7B-Chat",
    )
    parser.add_argument(
        "-f",
        "--sample-input-file",
        type=str,
        default=None,
        help="data path to HumanEval.jsonl",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="HumanEval_res.jsonl"
    )

    args = parser.parse_args()
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="auto",
        trust_remote_code=True,
        bf16=True,
        use_flash_attn=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))

    f = jsonlines.open(args.sample_input_file)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc="task_idx"):
            # use humanevalpack prompt
            signature = re.search(
                rf"def\s+({jobj['entry_point']}.*?):\s*\n", jobj["prompt"]
            ).group(1)
            description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", jobj["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
            prompt = (
                f"Write a Python function `{signature}` to solve the following problem:\n"
                f"{description}\n"
                f"{jobj['prompt']}"
            )

            task_id = jobj["task_id"]
            answer, response = generate_sample(
                model, tokenizer, prompt, jobj["entry_point"]
            )
            gen_jobjs = {"task_id": task_id, "completion": answer, "response": response}
            output.write(gen_jobjs)
    f_output.close()
