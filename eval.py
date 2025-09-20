import argparse
import io
import os
import tarfile
import subprocess
from pathlib import Path

MERGES_ROOT = Path("merges")
RESULTS_ROOT = Path("/results")


def is_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    has_cfg = (p / "config.json").exists()
    has_wts = any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin"))
    return has_cfg or has_wts


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


if __name__ == "__main__":
    main()
