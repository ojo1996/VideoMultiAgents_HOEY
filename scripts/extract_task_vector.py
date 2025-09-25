import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def load_state_dict(model_id_or_path: str, dtype: torch.dtype = torch.float32) -> Tuple[Dict[str, torch.Tensor], str]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    arch = model.config.model_type if hasattr(model, "config") else "unknown"
    sd = {k: v.detach().to(dtype).cpu() for k, v in model.state_dict().items()}
    # free model weights from memory
    del model
    return sd, arch


def check_compatible(base_arch: str, sft_arch: str, rl_arch: Optional[str]):
    if sft_arch != base_arch:
        raise ValueError(f"Base ({base_arch}) and SFT ({sft_arch}) architectures differ")
    if rl_arch is not None and rl_arch != base_arch:
        raise ValueError(f"RL ({rl_arch}) and Base ({base_arch}) architectures differ")


def compute_delta(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # returns (a - b) on intersecting keys; ignores buffers/shapes that mismatch
    out: Dict[str, torch.Tensor] = {}
    for k, ta in a.items():
        tb = b.get(k)
        if tb is None:
            continue
        if ta.shape != tb.shape:
            continue
        if not torch.is_floating_point(ta) or not torch.is_floating_point(tb):
            continue
        out[k] = (ta - tb).to(torch.float16)
    return out


def write_index(index_path: Path, entries: Dict[str, Dict]):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Extract task and reasoning vectors as weight deltas (safetensors fp16)")
    ap.add_argument("--base", required=True, help="HF model id or local path for base model")
    ap.add_argument("--sft", required=True, help="HF model id or local path for SFT model")
    ap.add_argument("--rl", default=None, help="Optional RL-tuned model (same family)")
    ap.add_argument("--out_root", default="vectors", help="Root output directory")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"], help="Load dtype for state dicts before diff")
    args = ap.parse_args()

    load_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    base_sd, base_arch = load_state_dict(args.base, dtype=load_dtype)
    sft_sd, sft_arch = load_state_dict(args.sft, dtype=load_dtype)
    rl_sd, rl_arch = (None, None)
    if args.rl:
        rl_sd, rl_arch = load_state_dict(args.rl, dtype=load_dtype)

    check_compatible(base_arch, sft_arch, rl_arch)

    # Task vector: tau_task = theta_SFT - theta_base
    tau_task = compute_delta(sft_sd, base_sd)

    # Reasoning vector (optional): v_reason = theta_RL - theta_SFT
    v_reason = None
    if rl_sd is not None:
        v_reason = compute_delta(rl_sd, sft_sd)

    out_root = Path(args.out_root)
    paths = {
        "index": out_root / "index.json",
        "tasks_dir": out_root / "tasks",
        "reason_dir": out_root / "reasoning",
    }
    paths["tasks_dir"].mkdir(parents=True, exist_ok=True)
    if v_reason is not None:
        paths["reason_dir"].mkdir(parents=True, exist_ok=True)

    task_file = paths["tasks_dir"] / "task_delta_fp16.safetensors"
    save_file(tau_task, str(task_file))

    reason_file = None
    if v_reason is not None:
        reason_file = paths["reason_dir"] / "reason_delta_fp16.safetensors"
        save_file(v_reason, str(reason_file))

    # Build index.json with per-tensor rows and which file they live in
    index_rows = {}
    for k, t in tau_task.items():
        index_rows[k] = {
            "where": str(task_file.relative_to(out_root)),
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "vector": "task",
        }
    if v_reason is not None:
        for k, t in v_reason.items():
            index_rows[k] = {
                "where": str(reason_file.relative_to(out_root)),
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
                "vector": "reason",
            }

    write_index(paths["index"], index_rows)

    print(f"[ok] wrote task vector to {task_file}")
    if reason_file:
        print(f"[ok] wrote reasoning vector to {reason_file}")
    print(f"[ok] wrote index to {paths['index']}")


if __name__ == "__main__":
    main()


