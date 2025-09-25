import argparse
import itertools
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Sweep alpha values and optionally evaluate")
    ap.add_argument("--base", required=True)
    ap.add_argument("--vector_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, required=True)
    ap.add_argument("--eval_task", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for a in args.alphas:
        tag = f"alpha={a}"
        out_dir = out_root / tag
        cmd = [
            "python", "scripts/apply_vector_and_eval.py",
            "--base", args.base,
            "--vector_root", args.vector_root,
            "--alpha_task", str(a),
            "--alpha_reason", "0.0",
            "--out_dir", str(out_dir),
        ]
        if args.eval_task:
            cmd += ["--eval_task", args.eval_task, "--device", args.device]
        print(f"[*] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    print("[ok] sweep complete")


if __name__ == "__main__":
    main()


