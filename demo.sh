#!/usr/bin/env bash
set -euo pipefail

########## USER CONFIG (edit if needed) ##########
BASE="Qwen/Qwen2.5-7B-Instruct"
SFT="models/AFM-CodeAgent-7B-sft"
RL="models/AFM-CodeAgent-7B-rl"

ALPHAS=("0.0" "0.5")          # alpha_task sweep (spaces ok) - super light
TASK="hellaswag"              # tiny eval task
DEVICE="mps"                  # use Apple GPU
RESULTS_ROOT="results"
MERGES_DIR="merges"
EVAL_CFG="configs/eval_debug.yaml"

# Ultra-light limits (keeps Mac runs super short)
DEBUG_LIMIT=10                # ~40 requests total
DEBUG_BATCH=4                 # smaller batch size
DEBUG_SEEDS="[1]"
##################################################

echo "== Safe profile for Apple Silicon =="
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4

# Check MPS availability (informative only)
python - <<'PY'
import torch, platform
print("arch:", platform.machine())
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY

# Ensure debug eval config exists (idempotent)
if [[ ! -f "$EVAL_CFG" ]]; then
  echo "== Creating $EVAL_CFG (debug profile) =="
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

echo "== Ultra-light HellaSwag eval on MPS (10 samples only) =="
python eval.py \
  --models "merges/alpha=0.0" "merges/alpha=0.5" \
  --results_root "$RESULTS_ROOT" \
  --config "$EVAL_CFG" \
  --task "$TASK" \
  --device "$DEVICE" \
  --alpha_json alpha_settings.json

# Optional: aggregate + plot (if you added these tools)
if [[ -f tools/aggregate.py ]]; then
  echo "== Aggregating results =="
  python tools/aggregate.py || true
fi
if [[ -f tools/plot_alpha.py ]]; then
  echo "== Plotting alpha curves =="
  python tools/plot_alpha.py || true
fi

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

echo "== Done (ultra-light demo complete) =="
echo "Results under: $RESULTS_ROOT/batch/<timestamp>/ and $RESULTS_ROOT/bash_probe/"
echo "Total requests: ~40 (10 samples × 2 models × 2 alphas)"
