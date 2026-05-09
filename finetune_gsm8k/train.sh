#!/bin/bash
# =============================================================================
# STEP 3: Training Script
# =============================================================================
# What this does:
#   Runs LoRA fine-tuning on Qwen-1.8B-Chat using GSM8K training data
#
# Why Qwen-1.8B-Chat:
#   Smallest model = least GPU memory needed. Good for learning.
#   "Chat" variant = already trained to follow instructions, fine-tunes better.
#
# Why LoRA:
#   LoRA (Low-Rank Adaptation) freezes 99% of model weights and only trains
#   small "adapter" layers. This means:
#     - 10x less GPU memory than full fine-tuning
#     - Much faster training
#     - Almost same quality improvement
#
# Hardware needed: at minimum 8GB GPU VRAM (RTX 3070, 4060 Ti, etc.)
# If no GPU: Use Google Colab (free) — see GUIDE.md
#
# Run this from the repo root:  bash finetune_gsm8k/train.sh
# =============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0  # Use first GPU. If you have multiple, can set to 0,1 etc.

# --- CONFIG (change these if needed) ---

MODEL="Qwen/Qwen-1_8B-Chat"   # Downloads automatically from HuggingFace
                                # Options: Qwen/Qwen-7B-Chat (needs ~20GB VRAM)
                                #          Qwen/Qwen-1_8B-Chat (needs ~7GB VRAM) ← recommended for beginners

DATA="finetune_gsm8k/train.json"
OUTPUT_DIR="finetune_gsm8k/output"

# --- TRAINING HYPERPARAMETERS explained ---
# num_train_epochs: How many times to go through all training data. 3-5 is typical.
# per_device_train_batch_size: How many examples per GPU step. Lower = less memory.
# gradient_accumulation_steps: Simulate larger batch by accumulating gradients.
#   Effective batch size = per_device_train_batch_size × gradient_accumulation_steps = 2×8 = 16
# learning_rate: How fast to update weights. 3e-4 is standard for LoRA.
# model_max_length: Max token length. GSM8K answers are short, 512 is enough.
# lora_r: LoRA rank. Higher = more parameters trained = more expressive but slower.
# lora_alpha: LoRA scaling. Usually set to lora_r or 2×lora_r.

mkdir -p $OUTPUT_DIR

python finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --bf16 True \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 3 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora \
  --lora_r 64 \
  --lora_alpha 16

echo ""
echo "============================================"
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Next step: run evaluate.sh or chat.py"
echo "============================================"
