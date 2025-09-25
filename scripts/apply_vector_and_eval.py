import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file as safe_load, save_file as safe_save
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_vectors(index_path: Path, root: Path) -> Dict[str, Dict]:
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    file_to_keys: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, meta in idx.items():
        rel = meta["where"]
        file_to_keys.setdefault(rel, {})[name] = None  # placeholder
    # read each safetensors once
    loaded_files: Dict[str, Dict[str, torch.Tensor]] = {}
    for rel in file_to_keys.keys():
        loaded_files[rel] = safe_load(str(root / rel))
    return {"index": idx, "files": loaded_files}


def compose_weights(
    base_sd: Dict[str, torch.Tensor],
    vectors: Optional[Dict[str, Dict]],
    alpha_task: float = 0.0,
    alpha_reason: float = 0.0,
    extra_alphas: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    out = {k: v.clone().to(torch.float16) if torch.is_floating_point(v) else v.clone() for k, v in base_sd.items()}
    if not vectors:
        return out

    idx = vectors["index"]
    files = vectors["files"]

    for name, meta in idx.items():
        rel = meta["where"]
        vec_kind = meta.get("vector", "task")
        if name not in files[rel]:
            continue
        delta = files[rel][name]
        if name not in out:
            continue
        if out[name].shape != delta.shape or not torch.is_floating_point(out[name]):
            continue
        coeff = 0.0
        if vec_kind == "task":
            coeff += alpha_task
        elif vec_kind == "reason":
            coeff += alpha_reason
        if extra_alphas and name in extra_alphas:
            coeff += float(extra_alphas[name])
        if coeff == 0.0:
            continue
        out[name] = (out[name].to(torch.float16) + coeff * delta.to(torch.float16)).to(out[name].dtype)
    return out


def save_merged(model_id_or_path: str, state: Dict[str, torch.Tensor], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[warn] missing keys: {len(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def maybe_eval(model_dir: Path, task: Optional[str], device: str = "cuda:0") -> int:
    if not task:
        return 0
    if task.upper() == "ALL":
        # Defer to config-driven batch runner; nothing to do here.
        return 0
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", "pretrained=.,dtype=float16,trust_remote_code=True",
        "--tasks", task,
        "--device", device,
        "--batch_size", "1",
        "--gen_kwargs", "max_gen_toks=2048",
        "--log_samples",
        "--output_path", str(model_dir),
        "--confirm_run_unsafe_code",
    ]
    proc = subprocess.run(cmd, cwd=model_dir)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Apply linear weight vectors to a base model and optionally evaluate")
    ap.add_argument("--base", required=True, help="HF id or local path for base model")
    ap.add_argument("--vector_root", required=True, help="Root folder that contains vectors/index.json")
    ap.add_argument("--alpha_task", type=float, default=0.0)
    ap.add_argument("--alpha_reason", type=float, default=0.0)
    ap.add_argument("--extra_alpha_json", default=None, help="Optional JSON mapping of tensor_name -> alpha")
    ap.add_argument("--out_dir", required=True, help="Where to save the merged model")
    ap.add_argument("--eval_task", default=None, help="Optional lm-eval task name (e.g., humaneval, aime25)")
    ap.add_argument("--device", default="cuda:0")
    # Optional: delegate to mergekit recipe via existing generate_merged_models.py
    ap.add_argument("--use_mergekit", action="store_true", help="Use generate_merged_models.py instead of inline merge")
    ap.add_argument("--mk_base", default=None, help="Base model for mergekit (HF id or local dir)")
    ap.add_argument("--mk_rl", default=None, help="RL model for mergekit")
    ap.add_argument("--mk_sft", default=None, help="SFT model for mergekit")
    ap.add_argument("--mk_dtype", default="float16")
    args = ap.parse_args()

    def write_run_card(target_dir: Path, card: Dict[str, object]):
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / "run_card.json", "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=2)

    # If mergekit mode requested, invoke the repo's generator and exit
    if args.use_mergekit:
        if not (args.mk_base and args.mk_rl and args.mk_sft):
            raise ValueError("--use_mergekit requires --mk_base, --mk_rl, --mk_sft")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python", "generate_merged_models.py",
            "--base_model", args.mk_base,
            "--rl_model", args.mk_rl,
            "--sft_model", args.mk_sft,
            "--out_root", str(out_dir),
            "--dtype", args.mk_dtype,
            "--alphas", str(args.alpha_task),
        ]
        print(f"[*] running mergekit: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(rc)
        # If an eval task is provided, try to evaluate the produced folder (named alpha{val})
        produced = out_dir / f"alpha={args.alpha_task}"
        # Write a minimal run_card for provenance
        write_run_card(produced, {
            "mode": "mergekit",
            "base": args.mk_base,
            "sft": args.mk_sft,
            "rl": args.mk_rl,
            "alpha_task": args.alpha_task,
            "dtype": args.mk_dtype,
        })
        target = produced if produced.exists() else out_dir
        eval_rc = maybe_eval(target, args.eval_task, device=args.device)
        if eval_rc == 0 and args.eval_task:
            print("[ok] evaluation finished successfully")
        elif args.eval_task:
            print(f"[warn] evaluation exited with code {eval_rc}")
        return

    base_model = args.base
    vector_root = Path(args.vector_root)
    index_path = vector_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found")

    vectors = load_vectors(index_path, vector_root)

    # load base state
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_sd = {k: v.detach().cpu() for k, v in base_model_obj.state_dict().items()}
    del base_model_obj

    extra_alphas = None
    if args.extra_alpha_json:
        with open(args.extra_alpha_json, "r", encoding="utf-8") as f:
            extra_alphas = json.load(f)

    merged = compose_weights(
        base_sd,
        vectors,
        alpha_task=args.alpha_task,
        alpha_reason=args.alpha_reason,
        extra_alphas=extra_alphas,
    )

    out_dir = Path(args.out_dir)
    save_merged(base_model, merged, out_dir)
    # Write run card for inline merge
    write_run_card(out_dir, {
        "mode": "inline_additive",
        "base": base_model,
        "vector_root": str(vector_root),
        "alpha_task": args.alpha_task,
        "alpha_reason": args.alpha_reason,
        "extra_alpha_json": args.extra_alpha_json or None,
    })
    print(f"[ok] saved merged model to {out_dir}")

    rc = maybe_eval(out_dir, args.eval_task, device=args.device)
    if rc == 0 and args.eval_task:
        print("[ok] evaluation finished successfully")
    elif args.eval_task:
        print(f"[warn] evaluation exited with code {rc}")


if __name__ == "__main__":
    main()


