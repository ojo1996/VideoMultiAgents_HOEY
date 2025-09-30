#!/usr/bin/env bash
set -euo pipefail

TOOL="${1:-bash}"
MODEL="${2:-Qwen/Qwen2.5-0.5B}"

python training/trl_train_tool_sft.py \
  --config "configs/trl_defaults.yaml" \
  --tool "$TOOL" \
  --model_name "$MODEL" \
  --data_root "data/sft" \
  --out_dir "runs/sft"
