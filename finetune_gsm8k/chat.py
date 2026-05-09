"""
STEP 5: Interactive Chat with Your Fine-Tuned Model
=====================================================
What this does:
  Lets you chat with your fine-tuned model interactively in the terminal.
  You can ask it math questions and see if it reasons through them correctly.

How to run (from repo root):
  # Chat with fine-tuned model:
  python finetune_gsm8k/chat.py --model finetune_gsm8k/output

  # Chat with base model (for comparison):
  python finetune_gsm8k/chat.py --model Qwen/Qwen-1_8B-Chat

Type 'quit' or 'exit' to stop.
Try asking: "If Tom has 15 apples and gives 3 to each of his 4 friends, how many does he have left?"
"""

import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


def load_model(model_path: str):
    print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        print("Detected LoRA adapter — loading base + adapter...")
        with open(adapter_config) as f:
            cfg = json.load(f)
        base = cfg["base_model_name_or_path"]
        print(f"  Base: {base}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base, device_map="auto", trust_remote_code=True
        ).eval()
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True
        ).eval()

    try:
        model.generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
    except Exception:
        pass

    print("Model loaded!\n")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    print("=" * 60)
    print("QWEN MATH ASSISTANT — Fine-tuned on GSM8K")
    print("=" * 60)
    print("Ask any math word problem. Type 'exit' to quit.")
    print("Tip: Ask step-by-step problems like the training data.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bye!")
            break

        # Prepend instruction to match training format
        prompt = f"Solve the following math problem step by step.\n\nProblem: {user_input}"

        response, history = model.chat(tokenizer, prompt, history=history)

        print(f"\nAssistant: {response}\n")
        print("-" * 40)


if __name__ == "__main__":
    main()
