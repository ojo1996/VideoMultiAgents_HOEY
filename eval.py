import subprocess
from pathlib import Path

MERGES_ROOT = Path("merges")

def is_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    has_cfg = (p / "config.json").exists()
    has_wts = any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin"))
    return has_cfg or has_wts

def main():
    if not MERGES_ROOT.exists():
        print(f"[err] {MERGES_ROOT} not found")
        return

    dirs = sorted(d for d in MERGES_ROOT.iterdir() if is_model_dir(d))
    if not dirs:
        print(f"[warn] no model folders found in {MERGES_ROOT}")
        return

    for d in dirs:
        print(f"\n=== evaluating: {d} ===")

        cmd = [
            "lm_eval", "--model", "hf",
            "--model_args", "pretrained=.,dtype=float16,trust_remote_code=True",
            "--tasks", "aime25",
            "--device", "cuda:0",
            "--batch_size", "1",
            "--gen_kwargs", "max_gen_toks=2048",
            "--log_samples",
            "--output_path", "results_sft_aime25.json",
        ]
        # run with cwd so outputs (results & logs) land inside each model dir
        res = subprocess.run(cmd, cwd=d)
        if res.returncode != 0:
            print(f"[warn] lm_eval failed in {d} (exit {res.returncode})")

if __name__ == "__main__":
    main()
