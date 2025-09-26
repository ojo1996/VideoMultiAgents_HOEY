#!/usr/bin/env python3
# Quick, model-only bash probe (no real shell exec). Produces a simple accuracy.
# Usage:
#   python tools/bash_probe.py --models merges/alpha=0.5 merges/alpha=0.5_alpha_bash=1.0 --device mps --out-root results
#
# Notes:
# - Loads the HF model in each folder with dtype fp16 (or bf16) and generates a single-line bash command.
# - Scoring = fraction of probes whose expected regex appears in the model output.
# - This is a smoke test to visualize α_bash "dialability" locally; use CUDA for full evals.

import argparse, json, os, re, time, math
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBES = [
    # (instruction, regex pattern expected in the bash command)
    ("Create a file named tmp.txt with the text hi",       r"echo\s+hi\s*>\s*tmp\.txt"),
    ("Append bar to tmp.txt",                              r"echo\s+bar\s*>>\s*tmp\.txt"),
    ("List all files in the current directory",           r"\bls\b(?!.*-R)"),
    ("List files including hidden ones",                   r"\bls\s+-a\b"),
    ("Make a folder named data",                           r"\bmkdir\s+(-p\s+)?data\b"),
    ("Search for the word foo in a.txt and print line numbers", r"\bgrep\s+-n\s+['\"]?foo['\"]?\s+a\.txt\b"),
    ("Count lines in tmp.txt",                             r"\bwc\s+-l\s+tmp\.txt\b"),
    ("Print the first 5 lines of a.txt",                   r"\bhead\s+-n\s*5\s+a\.txt\b"),
    ("Replace all 'cat' with 'dog' in pets.txt (in place)", r"\bsed\s+-i(?:\s+''|)\s+['\"]s/cat/dog/g['\"]\s+pets\.txt\b"),
    ("Show the current working directory",                 r"\bpwd\b"),
]

SYSTEM_INSTR = (
    "You are a coding assistant that outputs ONLY a single bash command for the user's request. "
    "No explanations, no prose, no code fences. If multiple commands are needed, chain them with '&&'."
)

def clean_cmd(text: str) -> str:
    t = text.strip()
    # strip code fences or accidental prose
    t = re.sub(r"^```(?:bash)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    # take first line only
    t = t.splitlines()[0].strip() if t else t
    # mild sanitization
    return t

def dangerous(cmd: str) -> bool:
    bad = [
        r"\brm\s+-rf\b", r":\s*(){", r"\bdd\s+if=", r"\bmkfs\.", r"\bforkbomb\b",
        r"\bshutdown\b", r"\breboot\b"
    ]
    return any(re.search(p, cmd) for p in bad)

def load_model(model_dir: str, device: str = "mps", dtype: str = "float16"):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map={"": device}
    )
    model.eval()
    return tok, model

@torch.inference_mode()
def ask_for_cmd(tok, model, prompt: str, device: str):
    # Simple instruction-following prompt: system + user → assistant
    text = f"<system>{SYSTEM_INSTR}</system>\n<user>{prompt}</user>\n<assistant>"
    inputs = tok(text, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_cmd(gen)

def run_probe(model_dir: str, device: str) -> Dict[str, Any]:
    tok, model = load_model(model_dir, device=device)
    results = []
    hits = 0
    for i, (q, pat) in enumerate(PROBES, 1):
        cmd = ask_for_cmd(tok, model, q, device)
        ok = bool(re.search(pat, cmd))
        unsafe = dangerous(cmd)
        hits += 1 if ok and not unsafe else 0
        results.append({
            "id": i, "prompt": q, "pattern": pat, "cmd": cmd, "match": ok, "unsafe": unsafe
        })
    acc = hits / len(PROBES)
    return {"accuracy": acc, "n": len(PROBES), "results": results}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="HF model folders (e.g., merges/alpha=0.5 ...)")
    ap.add_argument("--device", default="mps", help="mps | cpu | cuda:0")
    ap.add_argument("--out-root", default="results", help="where to write probe reports")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out_root = Path(args.out_root) / "bash_probe"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for mdl in args.models:
        t0 = time.time()
        rep = run_probe(mdl, args.device)
        rep["model"] = mdl
        rep["device"] = args.device
        rep["timestamp"] = int(time.time())
        out_path = out_root / (Path(mdl).name + "_bash_probe.json")
        out_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        dt = time.time() - t0
        summary[Path(mdl).name] = {"accuracy": rep["accuracy"], "seconds": round(dt, 2)}
        print(f"[ok] {mdl}: bash_probe_acc={rep['accuracy']:.2f} in {dt:.1f}s → {out_path}")

    # quick side-by-side print
    if len(summary) >= 1:
        print("\n=== Bash Probe Summary ===")
        for name, s in summary.items():
            print(f"{name:40s}  acc={s['accuracy']:.2f}  time={s['seconds']}s")

if __name__ == "__main__":
    main()
