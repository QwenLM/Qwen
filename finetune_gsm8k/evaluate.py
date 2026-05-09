"""
STEP 4: Evaluation Script
==========================
What this does:
  Tests your fine-tuned model on the GSM8K test set (1319 math problems).
  Compares its answers to the correct answers.
  Reports accuracy as a percentage.

What "accuracy" means here:
  If the model's final number matches the correct answer → correct.
  GSM8K uses "#### <number>" to mark the final answer in the ground truth.
  We extract the last number from the model's response and compare.

How to run (from repo root):
  # Test the fine-tuned model:
  python finetune_gsm8k/evaluate.py --model finetune_gsm8k/output

  # Test the base model (to see how much improvement):
  python finetune_gsm8k/evaluate.py --model Qwen/Qwen-1_8B-Chat

  # Faster test on just 100 examples:
  python finetune_gsm8k/evaluate.py --model finetune_gsm8k/output --max-samples 100
"""

import json
import re
import math
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


INVALID_ANS = "[invalid]"


def extract_final_number(text: str):
    """
    Extracts the last number from a string.

    GSM8K ground truth: "...total is 5+3=8\n#### 8"  → we want "8"
    Model output:       "...so the answer is 42."      → we want "42"

    We find all numbers in the text and return the last one.
    """
    pattern = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    last = matches[-1].group().replace(",", "").replace("+", "").strip()
    return last if last else None


def answers_match(model_output: str, ground_truth: str) -> bool:
    """Returns True if model's final number matches the correct answer."""
    gold = extract_final_number(ground_truth)
    pred = extract_final_number(model_output)

    if gold is None or pred is None:
        return False
    try:
        return math.isclose(float(eval(gold)), float(eval(pred)), rel_tol=0, abs_tol=1e-4)
    except Exception:
        return False


def load_model_and_tokenizer(model_path: str):
    """
    Loads either:
    - A base HuggingFace model (e.g., "Qwen/Qwen-1_8B-Chat")
    - A LoRA fine-tuned model from a local directory

    LoRA models are saved as adapter weights on top of the base model.
    We detect this by looking for "adapter_config.json" in the directory.
    """
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # Check if this is a LoRA adapter directory
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        print("Detected LoRA adapter. Loading base model + adapter...")
        with open(adapter_config) as f:
            cfg = json.load(f)
        base_model_name = cfg["base_model_name_or_path"]
        print(f"  Base model: {base_model_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights into base model for faster inference
        print("  LoRA weights merged into base model.")
    else:
        print("Loading model directly...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    try:
        model.generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
    except Exception:
        pass
    model.generation_config.do_sample = False  # Greedy decoding: pick the most likely token
    model.generation_config.repetition_penalty = 1.0

    return model, tokenizer


def ask_model(model, tokenizer, question: str) -> str:
    """Sends a question to the model and gets its response."""
    prompt = f"Solve the following math problem step by step.\n\nProblem: {question}"
    response, _ = model.chat(tokenizer, prompt, history=None)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to N samples (useful for quick testing)")
    parser.add_argument("--output-file", type=str, default="finetune_gsm8k/eval_results.jsonl",
                        help="Where to save detailed results")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    print("\nLoading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]

    if args.max_samples:
        test_data = test_data.select(range(args.max_samples))
        print(f"Evaluating on {args.max_samples} samples (subset mode)")
    else:
        print(f"Evaluating on all {len(test_data)} test samples")

    correct = 0
    total = 0
    results = []

    print("\nRunning evaluation... (this takes a while)\n")
    for example in tqdm(test_data):
        question = example["question"]
        ground_truth = example["answer"]

        model_output = ask_model(model, tokenizer, question)
        is_correct = answers_match(model_output, ground_truth)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "model_output": model_output,
            "correct": is_correct,
        })

    accuracy = correct / total * 100

    # Save detailed results
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print(f"RESULTS FOR: {args.model}")
    print("=" * 50)
    print(f"Correct : {correct} / {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nDetailed results saved to: {args.output_file}")
    print("=" * 50)
    print("\nContext: Qwen-1.8B-Chat baseline is ~35-40% on GSM8K.")
    print("After fine-tuning you should see improvement toward 50-60%+.")


if __name__ == "__main__":
    main()
