#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(pwd)

echo "[*] Setting up AFM-CodeAgent-7B environment"

python -m pip install --upgrade pip wheel setuptools
pip install "huggingface_hub[cli]>=0.23" accelerate transformers datasets \
            peft bitsandbytes einops

# clone & install deps
[[ -d lm-evaluation-harness ]] || git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
pip install -e ./lm-evaluation-harness

[[ -d mergekit ]] || git clone https://github.com/arcee-ai/mergekit.git
pip install -e ./mergekit

# model downloads
mkdir -p models

echo "[*] Downloading Qwen2.5-7B..."
huggingface-cli download Qwen/Qwen2.5-7B \
  --local-dir models/AFM-CodeAgent-7B-rl \
  --local-dir-use-symlinks False

echo "[*] Downloading AFM-CodeAgent-7B-sft..."
huggingface-cli download PersonalAILab/AFM-CodeAgent-7B-sft \
  --local-dir models/AFM-CodeAgent-7B-sft \
  --local-dir-use-symlinks False

echo "[*] Downloading AFM-CodeAgent-7B-rl..."
huggingface-cli download PersonalAILab/AFM-CodeAgent-7B-rl \
  --local-dir models/AFM-CodeAgent-7B-rl \
  --local-dir-use-symlinks False

echo "[*] Downloading Qwen2.5-7B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False

echo "[✓] Setup complete. Activate venv anytime with:"
