#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
HF_TOKEN="${HF_TOKEN:-}"   # assumes you exported this before running
PROJECT_DIR=$(pwd)

echo "[*] Setting up AFM-CodeAgent-7B environment in $PROJECT_DIR"
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
pip install "huggingface_hub[cli]>=0.23" accelerate transformers datasets \
            peft bitsandbytes einops

# clone & install deps
[[ -d lm-evaluation-harness ]] || git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
pip install -e ./lm-evaluation-harness

[[ -d mergekit ]] || git clone https://github.com/arcee-ai/mergekit.git
pip install -e ./mergekit

[[ -d VideoMultiAgents_HOEY ]] || git clone https://github.com/ojo1996/VideoMultiAgents_HOEY.git

# model downloads
mkdir -p models
echo "[*] Downloading AFM-CodeAgent-7B-sft..."
huggingface-cli download PersonalAILab/AFM-CodeAgent-7B-sft \
  --local-dir models/AFM-CodeAgent-7B-sft \
  --local-dir-use-symlinks False

echo "[*] Downloading AFM-CodeAgent-7B-rl..."
huggingface-cli download PersonalAILab/AFM-CodeAgent-7B-rl \
  --local-dir models/AFM-CodeAgent-7B-rl \
  --local-dir-use-symlinks False

echo "[✓] Setup complete. Activate venv anytime with:"
echo "source $PROJECT_DIR/.venv/bin/activate"
