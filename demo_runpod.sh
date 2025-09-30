#!/usr/bin/env bash
set -euo pipefail

########## USER CONFIG (edit if needed) ##########
BASE="Qwen/Qwen2.5-7B-Instruct"
SFT="models/AFM-CodeAgent-7B-sft"
RL="models/AFM-CodeAgent-7B-rl"

ALPHAS=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0")  # More alpha points for better curve
TASK="hellaswag"              # eval task
DEVICE="cuda:0"               # use CUDA GPU
RESULTS_ROOT="results"
MERGES_DIR="merges"
EVAL_CFG="configs/eval_runpod.yaml"

# RunPod-optimized limits
DEBUG_LIMIT=100               # More samples for better statistics
DEBUG_BATCH=8                 # Larger batch size for GPU
DEBUG_SEEDS="[1, 2, 3]"      # Multiple seeds for confidence intervals
##################################################

echo "== RunPod GPU Profile =="
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Check CUDA availability
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
PY

# Ensure RunPod eval config exists
if [[ ! -f "$EVAL_CFG" ]]; then
  echo "== Creating $EVAL_CFG (RunPod profile) =="
  mkdir -p "$(dirname "$EVAL_CFG")"
  cat > "$EVAL_CFG" <<EOF
seeds: $DEBUG_SEEDS
device: "$DEVICE"
limit: $DEBUG_LIMIT
batch_size: $DEBUG_BATCH
num_fewshot: 0
tasks:
  hellaswag:
    lm_eval_id: hellaswag
    metric: acc_norm
logging:
  keep_lm_eval_artifacts: true
EOF
else
  echo "== Using existing $EVAL_CFG =="
fi

echo "== Building MergeKit models (alpha sweep) =="
mkdir -p "$MERGES_DIR"
for a in ${ALPHAS[@]}; do
  a_trim=$(echo "$a" | xargs)   # trim whitespace
  echo "-- alpha_task=$a_trim"
  python scripts/apply_vector_and_eval.py \
    --use_mergekit \
    --mk_base "$BASE" \
    --mk_rl   "$RL" \
    --mk_sft  "$SFT" \
    --alpha_task "$a_trim" \
    --out_dir "$MERGES_DIR"
done

echo "== Generating alpha_settings.json =="
python tools/auto_make_alpha_settings.py

echo "== HellaSwag eval on CUDA (100 samples × 6 models × 3 seeds) =="
python eval.py \
  --models "merges/alpha=0.0" "merges/alpha=0.2" "merges/alpha=0.4" "merges/alpha=0.6" "merges/alpha=0.8" "merges/alpha=1.0" \
  --results_root "$RESULTS_ROOT" \
  --config "$EVAL_CFG" \
  --task "$TASK" \
  --device "$DEVICE" \
  --alpha_json alpha_settings.json

# Aggregate + plot
echo "== Aggregating results =="
python tools/aggregate.py || true

echo "== Plotting alpha curves =="
python tools/plot_alpha.py || true

echo "== Bash tool-vector dial (optional) =="
BASH_VEC="vectors/tools/bash/index.json"
if [[ -f "$BASH_VEC" ]]; then
  echo "-- Found $BASH_VEC; composing alpha_bash=1.0 on top of alpha=0.5"
  python scripts/apply_vector_and_eval.py \
    --base "$MERGES_DIR/alpha=0.5" \
    --vector_root vectors \
    --alpha_bash 1.0 \
    --out_dir "$MERGES_DIR/alpha=0.5_alpha_bash=1.0"

  echo "-- Running no-exec bash probe (string match)"
  python tools/bash_probe.py \
    --models "$MERGES_DIR/alpha=0.5" "$MERGES_DIR/alpha=0.5_alpha_bash=1.0" \
    --device "$DEVICE" \
    --out-root "$RESULTS_ROOT"
else
  echo "-- No bash vector at $BASH_VEC; skipping tool-vector demo."
  echo "   To enable: train tiny bash SFT → merge LoRA → extract vector to $BASH_VEC"
fi

echo "== Done (RunPod demo complete) =="
echo "Results under: $RESULTS_ROOT/batch/<timestamp>/ and $RESULTS_ROOT/bash_probe/"
echo "Total requests: ~1800 (100 samples × 6 models × 3 seeds)"
echo "Check results/plots/ for alpha curve visualizations"
