# sweep_merge.py
# pip install pyyaml
import argparse, subprocess, tempfile, shutil
from pathlib import Path
import yaml

def build_recipe(base_model, rl_model, sft_model, alpha, dtype="float16"):
    # minimal task_arithmetic recipe built programmatically
    return {
        "merge_method": "task_arithmetic",
        "dtype": dtype,
        "base_model": base_model,
        "models": [
            {"model": rl_model,  "parameters": {"weight": float(alpha)}},     # +alpha * θ_RL
            {"model": sft_model, "parameters": {"weight": -float(alpha)}},    # -alpha * θ_SFT
        ],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="(optional) starting YAML; if omitted we build from scratch")
    ap.add_argument("--out_root", default="merges")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--rl_model", required=True)
    ap.add_argument("--sft_model", required=True)
    ap.add_argument("--alphas", nargs="+", required=True, type=float)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="extra flags passed to mergekit-yaml, e.g. --cuda")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    # if a base config is provided, we’ll load and modify it; else we synthesize
    base_cfg = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)

    for a in args.alphas:
        out_dir = out_root / f"alpha{a}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if base_cfg:
            cfg = base_cfg.copy()
            cfg["merge_method"] = "task_arithmetic"
            cfg["dtype"] = args.dtype
            cfg["base_model"] = args.base_model
            cfg["models"] = [
                {"model": args.rl_model,  "parameters": {"weight": float(a)}},
                {"model": args.sft_model, "parameters": {"weight": -float(a)}},
            ]
        else:
            cfg = build_recipe(args.base_model, args.rl_model, args.sft_model, a, args.dtype)

        # write a per-alpha recipe next to outputs (nice for provenance)
        recipe_path = out_dir / "recipe.yaml"
        with open(recipe_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        cmd = ["mergekit-yaml", str(recipe_path), str(out_dir)]
        if args.extra:
            cmd += args.extra

        print(f"[merge] alpha={a} -> {out_dir}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[warn] merge failed for alpha={a} (exit {res.returncode})")

if __name__ == "__main__":
    main()
