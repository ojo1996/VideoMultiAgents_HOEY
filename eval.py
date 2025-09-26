"""
Eval runner wired to configs/eval.yaml and model folders.

Features:
- --task ALL reads tasks from configs/eval.yaml and runs each
- writes metrics.json at the model run directory, including alpha values if present
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import yaml
import tarfile

def run_lm_eval(model_path: Path, task: str, out_dir: Path, dtype: str, device: str,
                batch_size: str = "1", seed: Optional[int] = None,
                limit: Optional[int] = None, num_fewshot: Optional[int] = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained=.,dtype={dtype},trust_remote_code=True",
        "--tasks", task,
        "--device", device,
        "--batch_size", str(batch_size),
        "--log_samples",
        "--output_path", str(out_dir),
        "--confirm_run_unsafe_code",
    ]
    # MC tasks don't need generation; drop the huge token budget to avoid confusion
    # cmd += ["--gen_kwargs", "max_gen_toks=2048"]

    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    log_path = out_dir / f"lm_eval__{task}.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(" ".join(shlex.quote(x) for x in cmd) + "\n")
        proc = subprocess.run(cmd, cwd=model_path, stdout=lf, stderr=lf, text=True)
    return proc.returncode

def safe_rename(src: Path, dst: Path):
    if src.exists():
        if dst.exists():
            dst.unlink()
        src.rename(dst)

def parse_metrics_from_log(log_path: Path, task_name: str, metric_name: str) -> Optional[float]:
    """Parse metrics from lm-eval log file."""
    if not log_path.exists():
        return None
    
    try:
        log_content = log_path.read_text(encoding="utf-8")
        
        # Look for the results table in the log
        # Pattern: |hellaswag|      1|none  |     0|acc_norm|↑  | 0.68|±  |0.0952|
        lines = log_content.split('\n')
        
        for line in lines:
            if '|' in line and task_name in line and '|' in line:
                # Split by | and look for metric values
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 7:
                    # Look for the metric name in the line
                    if metric_name in line:
                        # Find the value (usually the 6th or 7th part)
                        for part in parts:
                            try:
                                # Try to extract a float value
                                if re.match(r'^\d+\.\d+$', part):
                                    return float(part)
                            except ValueError:
                                continue
        
        # Fallback: look for any numeric value in lines containing the task
        for line in lines:
            if task_name in line and '|' in line:
                # Extract all numbers from the line
                numbers = re.findall(r'\d+\.\d+', line)
                if numbers:
                    return float(numbers[0])
        
        return None
    except Exception as e:
        print(f"[warn] Failed to parse log {log_path}: {e}")
        return None


def run_eval_for_model(model_dir: Path, tasks: List[Dict[str, str]], dtype: str, device: str, out_root: Path, seeds: Optional[List[int]] = None,
                       global_limit: Optional[int] = None,
                       global_bs: Optional[int] = None,
                       global_fs: Optional[int] = None) -> Dict[str, float]:
    model_name = model_dir.name
    per_task_scores: Dict[str, float] = {}
    per_task_cis: Dict[str, List[float]] = {}
    
    for t in tasks:
        task_name   = t.get("name")
        metric_name = t.get("metric", "score")
        lm_id       = t.get("lm_eval_id", task_name)
        limit       = t.get("limit", global_limit)
        batch_size  = t.get("batch_size", global_bs or 1)
        fewshot     = t.get("num_fewshot", global_fs)

        print(f"[eval] {lm_id} → fewshot={fewshot} limit={limit} batch_size={batch_size}")

        rc = 0
        # If seeds provided, run once per seed, then average metric and compute CI
        if seeds:
            vals: List[float] = []
            for s in seeds:
                rc = run_lm_eval(model_dir, lm_id, out_root / model_name, dtype=dtype, device=device, seed=s,
                                 limit=limit, num_fewshot=fewshot, batch_size=str(batch_size))
                if rc != 0:
                    break
                results_path = (out_root / model_name) / "results.json"
                picked = None
                
                # Try to read from results.json first
                if results_path.exists():
                    try:
                        data = json.loads(results_path.read_text(encoding="utf-8"))
                        section = data.get("results", {}).get(task_name)
                        if isinstance(section, dict):
                            # Look for the specific metric name first, then fall back to any numeric value
                            if metric_name in section and isinstance(section[metric_name], (int, float)):
                                picked = float(section[metric_name])
                            else:
                                # Fallback: pick the first metric ending with ",none" or any numeric value
                                for k, v in section.items():
                                    if (k.endswith(",none") or k in ["acc", "acc_norm", "exact_match", "f1"]) and isinstance(v, (int, float)):
                                        picked = float(v)
                                        break
                    except Exception as e:
                        print(f"[warn] Failed to parse results.json: {e}")
                
                # If results.json parsing failed, try parsing from log file
                if picked is None:
                    log_path = (out_root / model_name) / f"lm_eval__{lm_id}.log"
                    picked = parse_metrics_from_log(log_path, lm_id, metric_name)
                
                if picked is not None:
                    vals.append(picked)
            if rc == 0 and vals:
                mean_score = sum(vals) / max(1, len(vals))
                per_task_scores[f"{task_name}"] = mean_score
                # Compute 95% CI (simple approximation)
                if len(vals) > 1:
                    import statistics
                    std_err = statistics.stdev(vals) / (len(vals) ** 0.5)
                    ci_margin = 1.96 * std_err  # 95% CI
                    per_task_cis[f"{task_name}"] = [max(0, mean_score - ci_margin), min(1, mean_score + ci_margin)]
                # Also store hierarchical key
                per_task_scores[f"{task_name}/{metric_name}"] = mean_score
            elif rc != 0:
                per_task_scores[f"{task_name}"] = float("nan")
            continue
        # single-run path
        rc = run_lm_eval(model_dir, lm_id, out_root / model_name, dtype=dtype, device=device,
                         limit=limit, num_fewshot=fewshot, batch_size=str(batch_size))
        if rc != 0:
            per_task_scores[f"{task_name}"] = float("nan")
            continue
        # try to read lm-eval results.json and extract the correct metric
        results_path = (out_root / model_name) / "results.json"
        score = float("nan")
        
        # Try to read from results.json first
        if results_path.exists():
            try:
                data = json.loads(results_path.read_text(encoding="utf-8"))
                section = data.get("results", {}).get(task_name)
                if isinstance(section, dict):
                    # Look for the specific metric name first, then fall back to any numeric value
                    if metric_name in section and isinstance(section[metric_name], (int, float)):
                        score = float(section[metric_name])
                    else:
                        # Fallback: pick the first metric ending with ",none" or any numeric value
                        for k, v in section.items():
                            if (k.endswith(",none") or k in ["acc", "acc_norm", "exact_match", "f1"]) and isinstance(v, (int, float)):
                                score = float(v)
                                break
            except Exception as e:
                print(f"[warn] Failed to parse results.json: {e}")
        
        # If results.json parsing failed, try parsing from log file
        if score == float("nan"):
            log_path = (out_root / model_name) / f"lm_eval__{lm_id}.log"
            score = parse_metrics_from_log(log_path, lm_id, metric_name) or float("nan")
        per_task_scores[f"{task_name}"] = score
        per_task_scores[f"{task_name}/{metric_name}"] = score
    
    # Store CI data for later use
    per_task_scores["_ci_data"] = per_task_cis
    return per_task_scores


def tar_results_folder(archive_path: Path, folder_to_tar: Path):
    # Create / overwrite tar.gz of the entire results directory
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        # Use arcname='results' so the tar unpacks to ./results/ rather than absolute paths
        tar.add(folder_to_tar, arcname="results")


def main():
    ap = argparse.ArgumentParser(description="Config-driven evaluator over model folders")
    ap.add_argument("--models", nargs="+", required=True, help="List of model folders to evaluate")
    ap.add_argument("--results_root", default="results", help="Root folder for outputs")
    ap.add_argument("--config", default="configs/eval.yaml", help="YAML config path")
    ap.add_argument("--task", default="ALL", help="Task name or ALL to use config list")
    ap.add_argument("--dtype", default=None, help="Override dtype (else from config)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha_json", default=None, help="Path to JSON containing alpha settings to embed in metrics.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    # New flat schema support: seeds/device at top-level or nested
    seeds = cfg.get("seeds") or cfg.get("eval", {}).get("seeds") or []
    device = args.device or cfg.get("device") or cfg.get("eval", {}).get("device", "cuda:0")
    dtype = args.dtype or cfg.get("dtype") or cfg.get("model", {}).get("dtype", "float16")
    
    # Extract global parameters for limits and batch sizes
    global_limit   = cfg.get("limit") or cfg.get("eval", {}).get("limit")
    global_bs      = cfg.get("batch_size") or cfg.get("eval", {}).get("batch_size")
    global_fs      = cfg.get("num_fewshot") or cfg.get("eval", {}).get("num_fewshot")
    # tasks can be a mapping or list
    tasks_block = cfg.get("tasks") or cfg.get("eval", {}).get("tasks", [])
    tasks_cfg: List[Dict[str, str]] = []
    if isinstance(tasks_block, dict):
        for name, spec in tasks_block.items():
            tasks_cfg.append({"name": name, **(spec or {})})
    else:
        tasks_cfg = tasks_block
    if args.task != "ALL":
        tasks = [t for t in tasks_cfg if t.get("name") == args.task]
    else:
        tasks = tasks_cfg

    results_root = Path(args.results_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = results_root / "batch" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    alpha_payload = None
    if args.alpha_json and Path(args.alpha_json).exists():
        alpha_payload = json.loads(Path(args.alpha_json).read_text(encoding="utf-8"))

    # Validate model folders and print discovered alphas
    model_dirs = []
    for model_dir_str in args.models:
        model_dir = Path(model_dir_str)
        if not model_dir.exists():
            print(f"[warn] model folder not found: {model_dir}")
            continue
        model_dirs.append(model_dir)
    
    if alpha_payload:
        print("[*] Discovered alphas:")
        for model_dir in model_dirs:
            alpha_info = alpha_payload.get(model_dir.name, {})
            print(f"  {model_dir.name}: {alpha_info}")
    
    # Create repro bundle
    repro_card = {
        "alpha_settings_json": "alpha_settings.json" if args.alpha_json else None,
        "eval_config": args.config,
        "git_commit": None,
        "models": [str(m) for m in model_dirs],
        "seeds": seeds,
        "device": device,
        "dtype": dtype,
    }
    try:
        import subprocess
        repro_card["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    
    repro_path = run_root / "repro_card.json"
    repro_path.write_text(json.dumps(repro_card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote repro bundle to {repro_path}")

    for model_dir in model_dirs:
        per_task = run_eval_for_model(model_dir, tasks, dtype=dtype, device=device, out_root=run_root, seeds=seeds,
                                      global_limit=global_limit,
                                      global_bs=global_bs,
                                      global_fs=global_fs)
        
        # Extract CI data and clean metrics
        ci_data = per_task.pop("_ci_data", {})
        clean_metrics = {k: v for k, v in per_task.items() if not k.startswith("_")}
        
        # Task coverage check: ensure every configured task appears in metrics
        expected_tasks = {t.get("name") for t in tasks}
        found_tasks = {k for k in clean_metrics.keys() if "/" not in k}  # flat keys only
        missing_tasks = expected_tasks - found_tasks
        if missing_tasks:
            print(f"[warn] missing tasks in {model_dir.name}: {missing_tasks}")
            # Add null entries for missing tasks
            for task in missing_tasks:
                clean_metrics[task] = None
        
        metrics_path = run_root / model_dir.name / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": (alpha_payload or {}).get(model_dir.name, alpha_payload or {}),
            "seeds": seeds,
            "metrics": clean_metrics,
            "metrics_ci": ci_data,
        }
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote {metrics_path}")

if __name__ == "__main__":
    main()
    