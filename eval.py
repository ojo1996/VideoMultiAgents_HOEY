"""
human_eval_runner.py

Run EleutherAI lm-evaluation-harness on HumanEval for multiple models,
using consistent parameters and writing each model's outputs into its
own subfolder under a shared results root.

Example:
  python human_eval_runner.py \
    --models merges/alpha-1.0 merges/alpha-2.0 /models/Qwen2.5-7B \
    --labels vec_a1 vec_a2 base \
    --results_root results \
    --dtype bfloat16 \
    --device cuda:0 \
    --temperature 0.5 \
    --max_gen_toks 2048 \
    --apply_chat_template \
    --system_instruction "Think step by step."
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

def run_humaneval_for_model(
    model_path: str,
    out_dir: Path,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    temperature: float = 0.5,
    max_gen_toks: int = 2048,
    batch_size: str = "1",
    apply_chat_template: bool = False,
    system_instruction: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> int:
    """
    Runs lm_eval on HumanEval for a single model checkpoint, writing outputs to out_dir.
    Returns the lm_eval process return code.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    output_json = out_dir / "results_humaneval.json"
    log_file = out_dir / "lm_eval_stdout_stderr.log"
    cmd_used_file = out_dir / "command.txt"

    base_cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype={dtype},trust_remote_code=True",
        "--tasks", "humaneval",
        "--device", device,
        "--batch_size", batch_size,
        "--gen_kwargs", f"temperature={temperature},max_gen_toks={max_gen_toks}",
        "--log_samples",
        "--output_path", str(output_json),
        "--confirm_run_unsafe_code"
    ]

    if apply_chat_template:
        base_cmd += ["--apply_chat_template", "True"]
    if system_instruction:
        base_cmd += ["--system_instruction", system_instruction]
    if extra_args:
        base_cmd += list(extra_args)

    # Save the exact command for provenance
    with open(cmd_used_file, "w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(part) for part in base_cmd) + "\n")

    # Run and tee output to a log file
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"[{datetime.now().isoformat()}] Running:\n")
        lf.write(" ".join(shlex.quote(p) for p in base_cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(base_cmd, stdout=lf, stderr=lf, text=True)

    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Batch runner for HumanEval with lm_eval across multiple models.")
    ap.add_argument("--models", nargs="+", required=True, help="List of model paths (dirs or HF IDs).")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels for each model (same length as --models).")
    ap.add_argument("--results_root", default="results", help="Root folder for all outputs.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"], help="Model dtype for loading.")
    ap.add_argument("--device", default="cuda:0", help="Device for evaluation, e.g., cuda:0 or cpu.")
    ap.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature for HumanEval (paper uses 0.5).")
    ap.add_argument("--max_gen_toks", type=int, default=2048, help="Max generated tokens.")
    ap.add_argument("--batch_size", default="1", help="Batch size (recommend 1 for reproducibility).")
    ap.add_argument("--apply_chat_template", action="store_true", help="Apply chat template (if models are Instruct/Chat).")
    ap.add_argument("--system_instruction", default=None, help='Optional system instruction, e.g., "Think step by step."')
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args appended to lm_eval (advanced use).")
    args = ap.parse_args()

    models = args.models
    labels = args.labels
    if labels and len(labels) != len(models):
        print("[error] --labels length must match --models length", file=sys.stderr)
        sys.exit(2)

    # Timestamps to separate runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.results_root) / "humaneval" / timestamp
    root.mkdir(parents=True, exist_ok=True)

    print(f"[*] Results root: {root}")

    # Iterate models
    for i, m in enumerate(models):
        label = labels[i] if labels else Path(m).name.replace("/", "_")
        out_dir = root / label
        print(f"[*] Running HumanEval for model[{i}] '{m}' -> {out_dir}")
        rc = run_humaneval_for_model(
            model_path=m,
            out_dir=out_dir,
            dtype=args.dtype,
            device=args.device,
            temperature=args.temperature,
            max_gen_toks=args.max_gen_toks,
            batch_size=args.batch_size,
            apply_chat_template=args.apply_chat_template,
            system_instruction=args.system_instruction,
            extra_args=args.extra,
        )
        status = "OK" if rc == 0 else f"EXIT {rc}"
        print(f"[+] Done: {label} ({status})")

    print("[*] All evaluations completed.")


if __name__ == "__main__":
    main()
