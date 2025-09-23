# run_mini_swebench.py
import os
import subprocess
import shlex
from pathlib import Path
from typing import Optional, List

def run_swebench_mini(
    model_name: str,
    subset: str = "verified",      # "lite" or "verified"
    split: str = "test",            # "dev" or "test"
    out_dir: str = "runs/mini_swebench",
    workers: int = 4,
    shuffle: bool = True,
    extra_config: Optional[str] = None,   # e.g. "swebench_roulette" if you want roulette mode
    instance_ids: Optional[List[str]] = None,  # run a few specific instances by id
    env: Optional[dict] = None,     # e.g. {"ANTHROPIC_API_KEY": "...", "OPENAI_API_KEY": "..."}
) -> None:
    """
    Launches mini-SWE-agent on SWE-bench via the official CLI helper and stores:
      - trajectories under <out_dir>/trajectories/*.traj.json
      - predictions in <out_dir>/preds.json
      - logs in <out_dir>/*
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "mini-extra", "swebench",
        "--subset", subset,
        "--split", split,
        "-o", out_dir,
        "--model", model_name,
        "--workers", str(workers),
    ]
    if shuffle:
        cmd.append("--shuffle")
    if extra_config:
        cmd += ["-c", extra_config]
    if instance_ids:
        # You can pass multiple --instance filters
        for iid in instance_ids:
            cmd += ["--instance", iid]

    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, env=({**env} if env else None))

if __name__ == "__main__":
  google_api_key = os.environ.get('GOOGLE_API_KEY')
  
  run_swebench_mini(
  model_name="google/gemini-2.5-pro-flash",
  subset="verified",
  split="test",
  out_dir="runs/gemini25proflash_verified_test",
  workers=8,
  shuffle=True,
  env={
      "GOOGLE_API_KEY": google_api_key
  },
)
