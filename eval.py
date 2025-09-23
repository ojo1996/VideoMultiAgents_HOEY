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

def safe_rename(src: Path, dst: Path):
    if src.exists():
        if dst.exists():
            dst.unlink()
        src.rename(dst)


def run_eval_for_model(model_dir: Path, dataset: str, logger):
    model_name = model_dir.name
    out_dir = RESULTS_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-model log file (captures both our logs and lm_eval stdout/stderr)
    log_path = out_dir / f"{model_name}__{dataset}.log"
    with log_path.open("ab", buffering=0) as logf:
        # Helper to write our own messages to the same log
        def log(msg: str):
            data = (msg.rstrip() + "\n").encode("utf-8", "replace")
            logf.write(data)

        log(f"\n=== evaluating: {model_dir} (dataset={dataset}) ===")

        # We set --output_path to out_dir so lm_eval drops its default files there.
        cmd = [
            "lm_eval", "--model", "hf",
            "--model_args", "pretrained=.,dtype=float16,trust_remote_code=True",
            "--tasks", dataset,
            "--device", "cuda:0",
            "--batch_size", "1",
            "--gen_kwargs", "max_gen_toks=2048",
            "--log_samples",
            "--output_path", str(out_dir),
            "--confirm_run_unsafe_code",
        ]

        log(f"[cmd] {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=model_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True
        )

        # Stream lm_eval output line-by-line into the log
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line.encode("utf-8", "replace"))

        rc = proc.wait()
        if rc != 0:
            log(f"[warn] lm_eval failed in {model_dir} (exit {rc})")
            return rc

        # Rename results & samples sensibly
        # lm-eval typically writes results.json and samples*.jsonl inside output_path.
        results_src = out_dir / "results.json"
        samples_src_candidates = list(out_dir.glob("*samples*.jsonl")) or list(out_dir.glob("samples.jsonl"))

        results_dst = out_dir / f"{model_name}__{dataset}__results.json"
        samples_dst = out_dir / f"{model_name}__{dataset}__samples.jsonl"

        if results_src.exists():
            safe_rename(results_src, results_dst)
            log(f"[ok] results -> {results_dst.name}")
        else:
            log("[warn] results.json not found; skipping rename")

        if samples_src_candidates:
            # If multiple, pick the largest (most complete) file
            samples_src = max(samples_src_candidates, key=lambda p: p.stat().st_size)
            safe_rename(samples_src, samples_dst)
            log(f"[ok] samples -> {samples_dst.name}")
        else:
            log("[warn] samples jsonl not found; skipping rename")

        log("[done] evaluation finished successfully")
        return 0


def tar_results_folder(archive_path: Path, folder_to_tar: Path):
    # Create / overwrite tar.gz of the entire results directory
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        # Use arcname='results' so the tar unpacks to ./results/ rather than absolute paths
        tar.add(folder_to_tar, arcname="results")


def main():
    parser = argparse.ArgumentParser(description="Evaluate merged models with lm-eval")
    parser.add_argument(
        "--dataset",
        "-d",
        default="aime25",
        help="Dataset/task name for lm-eval"
    )
    args = parser.parse_args()
    dataset = args.dataset

    if not MERGES_ROOT.exists():
        print(f"[err] {MERGES_ROOT} not found")
        return

    model_dirs = sorted(d for d in MERGES_ROOT.iterdir() if is_model_dir(d))
    if not model_dirs:
        print(f"[warn] no model folders found in {MERGES_ROOT}")
        return

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Overall run log
    run_log = RESULTS_ROOT / f"_run__{dataset}.log"
    with run_log.open("a", encoding="utf-8") as overall_log:
        for d in model_dirs:
            rc = run_eval_for_model(d, dataset, overall_log)
            if rc != 0:
                print(f"[warn] lm_eval failed in {d} (exit {rc})")

    # Tarball the entire /results directory
    archive = Path("/results.tar.gz")
    tar_results_folder(archive, RESULTS_ROOT)
    print(f"[ok] archived results to: {archive}")
    
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
    