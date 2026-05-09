"""
STEP 2: Data Preparation Script
================================
What this does:
  - Downloads the GSM8K dataset from HuggingFace (automatically, no manual download needed)
  - Converts it into the format that Qwen's finetune.py expects
  - Saves train.json and test.json

Why we need this:
  Qwen's training script expects data in a specific "conversations" format.
  GSM8K data looks like: {"question": "...", "answer": "... #### 42"}
  We need to convert it to: {"conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]}

Run this with:  python finetune_gsm8k/prepare_data.py
"""

import json
import re
from datasets import load_dataset

def extract_clean_answer(raw_answer: str) -> str:
    """
    GSM8K answers look like:
      "She has 3 apples. She buys 2 more. 3+2=5\n#### 5"

    The '####' marks the final numeric answer.
    We keep the full reasoning chain (chain-of-thought) so the model learns to THINK step by step.
    """
    return raw_answer.strip()

def convert_to_qwen_format(example: dict) -> dict:
    """
    Converts one GSM8K example to Qwen's conversation format.

    Qwen expects:
    {
        "id": "some_id",
        "conversations": [
            {"from": "user",      "value": "the math question"},
            {"from": "assistant", "value": "step-by-step reasoning + final answer"}
        ]
    }

    We add a system-level instruction in the user message to encourage step-by-step thinking.
    """
    question = example["question"].strip()
    answer   = extract_clean_answer(example["answer"])

    return {
        "conversations": [
            {
                "from": "user",
                "value": f"Solve the following math problem step by step.\n\nProblem: {question}"
            },
            {
                "from": "assistant",
                "value": answer
            }
        ]
    }

def main():
    print("=" * 60)
    print("Downloading GSM8K dataset from HuggingFace...")
    print("(This may take a minute on first run)")
    print("=" * 60)

    # load_dataset automatically downloads and caches the dataset
    # GSM8K has two splits: "train" (7473 examples) and "test" (1319 examples)
    dataset = load_dataset("gsm8k", "main")

    train_data = [convert_to_qwen_format(ex) for ex in dataset["train"]]
    test_data  = [convert_to_qwen_format(ex) for ex in dataset["test"]]

    with open("finetune_gsm8k/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open("finetune_gsm8k/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Training examples : {len(train_data)}")
    print(f"  Test examples     : {len(test_data)}")
    print(f"\nFiles saved:")
    print(f"  finetune_gsm8k/train.json")
    print(f"  finetune_gsm8k/test.json")
    print("\nSample training example:")
    print(json.dumps(train_data[0], indent=2))

if __name__ == "__main__":
    main()
