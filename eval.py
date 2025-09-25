"""
Eval runner wired to configs/eval.yaml and model folders.

Features:
- --task ALL reads tasks from configs/eval.yaml and runs each
- writes metrics.json at the model run directory, including alpha values if present
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
from typing import List, Optional, Dict

import yaml
import tarfile

def run_lm_eval(model_path: Path, task: str, out_dir: Path, dtype: str, device: str, batch_size: str = "1", seed: Optional[int] = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", "pretrained=.,dtype=%s,trust_remote_code=True" % dtype,
        "--tasks", task,
        "--device", device,
        "--batch_size", batch_size,
        "--gen_kwargs", "max_gen_toks=2048",
        "--log_samples",
        "--output_path", str(out_dir),
        "--confirm_run_unsafe_code",
    ]
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


def run_eval_for_model(model_dir: Path, tasks: List[Dict[str, str]], dtype: str, device: str, out_root: Path, seeds: Optional[List[int]] = None) -> Dict[str, float]:
    model_name = model_dir.name
    per_task_scores: Dict[str, float] = {}
    per_task_cis: Dict[str, List[float]] = {}
    
    for t in tasks:
        task_name = t.get("name")
        metric_name = t.get("metric", "score")  # Default metric name
        rc = 0
        # If seeds provided, run once per seed, then average metric and compute CI
        if seeds:
            vals: List[float] = []
            for s in seeds:
                rc = run_lm_eval(model_dir, task_name, out_root / model_name, dtype=dtype, device=device, seed=s)
                if rc != 0:
                    break
                results_path = (out_root / model_name) / "results.json"
                if results_path.exists():
                    data = json.loads(results_path.read_text(encoding="utf-8"))
                    section = data.get("results", {}).get(task_name)
                    if isinstance(section, dict):
                        # pick the first metric ending with ",none"
                        picked = None
                        for k, v in section.items():
                            if k.endswith(",none") and isinstance(v, (int, float)):
                                picked = float(v)
                                break
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
        rc = run_lm_eval(model_dir, task_name, out_root / model_name, dtype=dtype, device=device)
        if rc != 0:
            per_task_scores[f"{task_name}"] = float("nan")
            continue
        # try to read lm-eval results.json and take the first metric ending with ",none"
        results_path = (out_root / model_name) / "results.json"
        score = float("nan")
        if results_path.exists():
            data = json.loads(results_path.read_text(encoding="utf-8"))
            section = data.get("results", {}).get(task_name)
            if isinstance(section, dict):
                for k, v in section.items():
                    if k.endswith(",none") and isinstance(v, (int, float)):
                        score = float(v)
                        break
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
        per_task = run_eval_for_model(model_dir, tasks, dtype=dtype, device=device, out_root=run_root, seeds=seeds)
        
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
    