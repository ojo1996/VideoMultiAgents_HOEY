import json, glob, yaml
import sys
from pathlib import Path


def read(p):
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return None


def main(tol: float = 0.05, cfg_path: str = None):
    # Load per-task tolerances if config provided
    task_tols = {}
    if cfg_path and Path(cfg_path).exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        default_tol = cfg.get("default_tol", tol)
        task_tols = cfg.get("tasks", {})
    else:
        default_tol = tol

    base = None
    one = None
    for j in glob.glob("results/batch/*/alpha=*/metrics.json"):
        d = read(j)
        if not d:
            continue
        a = (d.get("alpha") or {}).get("alpha_task")
        if a == 0.0:
            base = d["metrics"]
        if a == 1.0:
            one = d["metrics"]
    if base is None or one is None:
        print("[warn] missing alpha=0.0 or alpha=1.0 in results; skipping check")
        return 0
    bad = []
    for k, v in one.items():
        b = base.get(k)
        if b is None:
            continue
        # Use per-task tolerance or fall back to default
        task_tol = task_tols.get(k, default_tol)
        if abs(v - b) > task_tol:
            bad.append((k, b, v))
    if bad:
        print("[fail] anchor check exceeded tolerance:")
        for k, b, v in bad:
            print(f"  {k}: alpha=0.0 {b} vs alpha=1.0 {v} (tol: {task_tols.get(k, default_tol)})")
        return 2
    print("[ok] anchor check passed within tolerance")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05, help="Default tolerance")
    ap.add_argument("--cfg", default=None, help="YAML config with per-task tolerances")
    args = ap.parse_args()
    raise SystemExit(main(args.tol, args.cfg))


