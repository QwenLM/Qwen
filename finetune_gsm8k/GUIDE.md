# Complete Beginner Guide: Fine-Tuning Qwen on GSM8K

This guide walks you through everything — from zero to a working fine-tuned math model.
No prior ML experience needed.

---

## What Are We Doing and Why?

### The Big Picture

**Large Language Models (LLMs)** like Qwen are trained on massive amounts of text from the internet.
They learn general language skills but aren't specialists at anything.

**Fine-tuning** means taking that pre-trained model and giving it extra training on a specific dataset
so it gets better at a specific task. It's like hiring a generalist employee and then training them
specifically for your job.

**GSM8K** (Grade School Math 8K) is a dataset of 8,500 grade school math word problems created by
OpenAI. It tests whether a model can reason through multi-step math problems. Example:

> "Lisa has 3 apples. She buys 2 bags, each containing 5 apples. How many apples does she have?"
> Answer: 3 + (2 × 5) = 13

After fine-tuning, our model should be significantly better at solving these kinds of problems.

### Why Qwen-1.8B-Chat?

We're using the **1.8 billion parameter** version because:
- Small enough to run on a consumer GPU (8GB VRAM minimum)
- Fast to train (hours, not days)
- Still capable enough to learn math reasoning
- The "Chat" version is already trained to follow instructions, so it fine-tunes better

For comparison: GPT-4 has ~1.8 **trillion** parameters. We're starting small and that's smart.

### Why LoRA instead of full fine-tuning?

**Full fine-tuning** updates every single weight in the model (1.8 billion numbers).
You'd need ~76GB of GPU memory for the 7B model — way too expensive.

**LoRA** (Low-Rank Adaptation) works differently:
- Freezes 99% of the original model weights (they don't change)
- Adds tiny "adapter" layers (only ~1-5M extra parameters)
- Only trains those adapter layers
- Result: 10x less memory, 5x faster, ~same quality improvement

Think of it as adding sticky notes to a book instead of rewriting the whole book.

---

## Part 1: What IDE Should I Use?

**Use VS Code** (Visual Studio Code). It's free, widely used, and has excellent Python support.

### Setting Up VS Code

1. **Download VS Code**: https://code.visualstudio.com/
2. **Install these extensions** (click Extensions icon in left sidebar, search each):
   - `Python` (by Microsoft) — essential
   - `Pylance` (by Microsoft) — code intelligence
   - `Jupyter` (by Microsoft) — for notebooks
   - `GitLens` — helpful for git

3. **Open the project folder**:
   - File → Open Folder → select the `Qwen` directory you downloaded

4. **Select Python interpreter**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Choose your conda or venv environment (we'll create one below)

---

## Part 2: Hardware Requirements

### Do You Have a GPU?

Run this in your terminal to check:
```bash
nvidia-smi
```

If you see a GPU listed with VRAM info → great, use it.
If you get "command not found" → you have no NVIDIA GPU.

### GPU Requirements

| Your GPU VRAM | Can Run |
|--------------|---------|
| 8GB (RTX 3070, 4060 Ti, etc.) | Qwen-1.8B with LoRA ✓ |
| 12-16GB (RTX 3080, 4070, etc.) | Qwen-1.8B LoRA ✓, Qwen-7B Q-LoRA ✓ |
| 24GB (RTX 3090, 4090, etc.) | Qwen-7B LoRA ✓, Qwen-14B Q-LoRA ✓ |
| No GPU / weak GPU | Use Google Colab (see below) |

### No GPU? Use Google Colab (Free!)

Google Colab gives you free GPU access in your browser. No installation needed.

1. Go to https://colab.research.google.com/
2. Click "New notebook"
3. Go to Runtime → Change runtime type → GPU (T4 is free, gives you 16GB)
4. Clone the repo and follow the steps below in Colab cells

Google Colab alternative: **Kaggle Notebooks** (also free, gives 30hrs/week of GPU)
- https://www.kaggle.com/code
- Settings → Accelerator → GPU T4 x2

---

## Part 3: Environment Setup

### Step 1: Install Conda (if you don't have it)

Conda manages Python environments so different projects don't conflict with each other.

Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Pick the installer for your OS
- Run the installer, follow the prompts

### Step 2: Create a Virtual Environment

Open a terminal (in VS Code: Terminal → New Terminal), then:

```bash
# Create a new environment called "qwen-finetune" with Python 3.10
conda create -n qwen-finetune python=3.10 -y

# Activate it (you'll need to do this every time you open a new terminal)
conda activate qwen-finetune
```

**Why virtual environments?** They isolate your project's packages from other Python projects.
Like separate containers for different experiments.

### Step 3: Install PyTorch

PyTorch is the deep learning framework everything runs on.

**First, check your CUDA version** (if you have an NVIDIA GPU):
```bash
nvidia-smi
# Look for "CUDA Version: XX.X" in the top right
```

Then install the matching PyTorch:

```bash
# CUDA 12.1 (most common on newer drivers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# No GPU (CPU only, very slow for training but works for inference)
pip install torch torchvision torchaudio
```

### Step 4: Install Project Dependencies

Navigate to the Qwen directory, then:

```bash
cd /path/to/Qwen  # replace with your actual path

# Install base Qwen requirements
pip install -r requirements.txt

# Install fine-tuning extras
pip install -r finetune_gsm8k/requirements_finetune.txt
```

### Step 5: HuggingFace Account (for downloading models)

Models are downloaded from HuggingFace Hub automatically when you run training.
Some models require accepting a license. For Qwen models:

1. Create account at https://huggingface.co/
2. Log in via terminal:
```bash
pip install huggingface_hub
huggingface-cli login
# Paste your HuggingFace token (from https://huggingface.co/settings/tokens)
```

---

## Part 4: Running Everything

### The Full Workflow (4 Steps)

```
[Prepare Data] → [Train Model] → [Evaluate] → [Chat with Model]
      ↓                ↓              ↓               ↓
 train.json      output/dir     accuracy %      interactive
```

### Step 1: Navigate to Project Root

```bash
cd /path/to/Qwen  # the directory containing finetune.py
conda activate qwen-finetune
```

### Step 2: Prepare the Data

This downloads GSM8K and converts it to Qwen's format.

```bash
python finetune_gsm8k/prepare_data.py
```

**What happens:**
- Downloads GSM8K dataset (~5MB) from HuggingFace
- Creates `finetune_gsm8k/train.json` (7,473 examples)
- Creates `finetune_gsm8k/test.json` (1,319 examples)

**Expected output:**
```
Downloading GSM8K dataset from HuggingFace...
Done!
  Training examples : 7473
  Test examples     : 1319
Files saved:
  finetune_gsm8k/train.json
  finetune_gsm8k/test.json
```

### Step 3: Fine-Tune the Model

```bash
bash finetune_gsm8k/train.sh
```

**What happens:**
- Downloads Qwen-1.8B-Chat model (~3.6GB) from HuggingFace on first run
- Starts LoRA fine-tuning for 3 epochs
- Saves checkpoints to `finetune_gsm8k/output/`

**What you'll see:**
```
trainable params: 9,175,040 || all params: 1,834,356,736 || trainable%: 0.5001
...
{'loss': 1.234, 'learning_rate': 0.00029, 'epoch': 0.05}
{'loss': 1.156, 'learning_rate': 0.00028, 'epoch': 0.10}
...
```

The loss number should go DOWN over time. That means the model is learning.
If it's not going down, something is wrong.

**How long does this take?**
- RTX 3080 (16GB): ~2-3 hours for 3 epochs
- RTX 4090 (24GB): ~1-1.5 hours
- Google Colab T4 (16GB): ~3-4 hours

**Out of memory error?**
If you see "CUDA out of memory", edit `train.sh` and:
- Reduce `per_device_train_batch_size` from 2 to 1
- Reduce `model_max_length` from 512 to 256

### Step 4: Test the Base Model First (Baseline)

Before testing the fine-tuned model, check how the BASE model performs.
This gives you a comparison point.

```bash
# Quick test on 50 examples (fast, ~5 mins)
python finetune_gsm8k/evaluate.py \
  --model Qwen/Qwen-1_8B-Chat \
  --max-samples 50
```

Note the accuracy. Then test your fine-tuned model:

### Step 5: Evaluate the Fine-Tuned Model

```bash
# Quick test first (50 examples)
python finetune_gsm8k/evaluate.py \
  --model finetune_gsm8k/output \
  --max-samples 50

# Full test (all 1319 examples, takes ~30-60 mins)
python finetune_gsm8k/evaluate.py \
  --model finetune_gsm8k/output
```

**Expected results:**

| Model | GSM8K Accuracy |
|-------|---------------|
| Qwen-1.8B-Chat (base) | ~35-40% |
| After 3 epochs LoRA on GSM8K | ~55-65% |

If your fine-tuned model is better → fine-tuning worked!

### Step 6: Chat with Your Model

```bash
python finetune_gsm8k/chat.py --model finetune_gsm8k/output
```

Try asking it problems like:
- "James has 50 apples. He gives 1/5 to his sister and 1/2 of the rest to his friend. How many does he have left?"
- "A train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours. What is the total distance?"

Compare responses with:
```bash
python finetune_gsm8k/chat.py --model Qwen/Qwen-1_8B-Chat
```

---

## Part 5: Understanding What Just Happened

### The Training Loop (Simplified)

```
For each batch of training examples:
  1. Feed question → model generates an answer
  2. Compare model's answer to correct answer → calculate "loss" (how wrong it is)
  3. Backpropagate: figure out which weights caused the error
  4. Update those weights slightly to reduce the error
  5. Repeat ~1000+ times
```

After thousands of iterations, the model has "seen" many math problems and learned patterns:
- How to break problems into steps
- How to perform arithmetic operations
- The format of chain-of-thought reasoning

### What is the Loss?

Loss is a number measuring how wrong the model's predictions are.
- High loss (1.5+) = model is confused, making big errors
- Low loss (0.5-) = model is predicting well

During training you want to see loss decrease. If it plateaus, the model has learned what it can.

### What is Overfitting?

If you train for too many epochs, the model memorizes the training data instead of learning
general math reasoning. Signs:
- Loss on training data keeps going down
- But accuracy on test data stops improving or gets worse

3 epochs is usually safe for this task.

---

## Part 6: Troubleshooting

### "ModuleNotFoundError: No module named 'peft'"
```bash
pip install peft
```

### "CUDA out of memory"
Edit `finetune_gsm8k/train.sh`:
```bash
--per_device_train_batch_size 1 \  # reduced from 2
--model_max_length 256 \           # reduced from 512
```

### "Killed" during training (Linux, no CUDA error)
Your system ran out of RAM. Close other programs. Or reduce `model_max_length`.

### Model downloads very slowly
HuggingFace servers can be slow. The 1.8B model is ~3.6GB. Just wait.
Or use a mirror: set environment variable `HF_ENDPOINT=https://hf-mirror.com`

### Training loss is NaN
Usually a learning rate issue. Edit `train.sh` and reduce:
```bash
--learning_rate 1e-4 \  # reduced from 3e-4
```

### LoRA adapter not merging in evaluate.py
Make sure `peft` is installed: `pip install peft>=0.7.0`

---

## Part 7: What's Next?

Once you've done this successfully, you can explore:

1. **Try a larger model**: Change `MODEL="Qwen/Qwen-7B-Chat"` in `train.sh`
   (needs 20GB+ VRAM or Q-LoRA)

2. **Train longer**: Increase `num_train_epochs` from 3 to 5

3. **Try GPQA**: Graduate-level science questions. Harder dataset, same process.
   Change the data preparation to load GPQA instead of GSM8K.
   Dataset: `Idavidrein/gpqa` on HuggingFace

4. **Use Weights & Biases for tracking**: Change `--report_to "none"` to `--report_to "wandb"`
   after `pip install wandb && wandb login`

5. **Try Qwen2 or Qwen2.5**: Newer, better models from Alibaba:
   - `Qwen/Qwen2.5-1.5B-Instruct` (same size, much better)
   - Works with same scripts, just change the model name

---

## File Structure Reference

```
Qwen/
├── finetune.py                    ← Qwen's training script (don't modify)
├── finetune/
│   ├── ds_config_zero2.json       ← DeepSpeed config (memory optimization)
│   └── finetune_lora_single_gpu.sh
└── finetune_gsm8k/               ← Our custom project
    ├── GUIDE.md                   ← This file
    ├── prepare_data.py            ← Download + convert GSM8K (Step 2)
    ├── train.sh                   ← Run training (Step 3)
    ├── evaluate.py                ← Test model accuracy (Step 4)
    ├── chat.py                    ← Interactive chat (Step 5)
    ├── requirements_finetune.txt  ← All dependencies
    ├── train.json                 ← Created by prepare_data.py
    ├── test.json                  ← Created by prepare_data.py
    └── output/                    ← Created by train.sh (LoRA weights)
```
