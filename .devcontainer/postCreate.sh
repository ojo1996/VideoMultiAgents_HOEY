#!/usr/bin/env bash
set -euo pipefail

# 0) optional: your repo's requirements
if [[ -f "requirements.txt" ]]; then
  pip install -r requirements.txt
fi

# 1) repos you asked for (editable installs for harness + mergekit)
if [[ ! -d "lm-evaluation-harness" ]]; then
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
fi
pip install -e ./lm-evaluation-harness

if [[ ! -d "mergekit" ]]; then
  git clone https://github.com/arcee-ai/mergekit.git
fi
pip install -e ./mergekit

# 2) clone VideoMultiAgents_HOEY
if [[ ! -d "VideoMultiAgents_HOEY" ]]; then
  git clone https://github.com/ojo1996/VideoMultiAgents_HOEY.git
fi

# 3) model downloads (AFM-CodeAgent-7B only)
mkdir -p models

echo "Downloading AFM-CodeAgent-7B-sft ..."
huggingface-cli download PersonalAILab/AFM-CodeAgent-7B-sft \
  --local-dir models/AFM-CodeAgent-7B-sft \
  --local-dir-use-symlinks False

echo "Downloading AFM-CodeAgent-7B-rl ..."
huggingface-cli download PersonalAILab/AFM-CodeAgent-7B-rl \
  --local-dir models/AFM-CodeAgent-7B-rl \
  --local-dir-use-symlinks False

python - <<'PY'
import torch, os
print("Torch CUDA available:", torch.cuda.is_available(), "CUDA", torch.version.cuda)
print("GPUs:", torch.cuda.device_count())
print("HF caches:", os.environ.get("HF_HOME"))
PY

echo "Post-create done. If private models are involved, ensure HF_TOKEN was set on host."
