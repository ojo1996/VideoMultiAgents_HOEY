# scripts/build_sft_datasets.py
import argparse, json, glob, os, random
from typing import Dict, Any, List
import yaml
import re

def load_templates(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_traj(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def first_bash_block(text: str) -> str:
    m = re.search(r"```bash\s*(.*?)```", text or "", flags=re.S|re.I)
    return (m.group(1).strip() if m else "").strip()

def render_instruction(tpl: str, traj: Dict[str, Any], action: Dict[str, Any]) -> str:
    return tpl.format(
        question = traj.get("question",""),
        context  = "",  # add stitched context later if you store it
        notes    = "",  # for MHQA hops, collect prior notes if present
        path     = (action.get("input",{}) or {}).get("path",""),
    ).strip()

def get_response(action: Dict[str, Any], key: str) -> str:
    # key options: "action_output", "action_input.command"
    if key == "action_output":
        out = action.get("output", "")
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    if key == "action_input.command":
        return (action.get("input",{}) or {}).get("command","")
    return ""

def postprocess(resp: str, pp: Dict[str, Any]) -> str:
    if not pp:
        return resp
    if pp.get("keep_first_bash_block"):
        return first_bash_block(resp)
    return resp

def to_examples(traj: Dict[str, Any], templates: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in traj.get("actions", []):
        tool = a.get("tool")
        spec = templates.get("tools", {}).get(tool)
        if not spec:
            continue
        instr = render_instruction(spec["instruction"], traj, a)
        resp  = get_response(a, spec.get("response_from","action_output"))
        resp  = postprocess(resp, spec.get("postprocess", {}))
        if not resp:
            continue
        out.append({
            "tool": tool,
            "instruction": instr,
            "output": resp,
            "meta": {
                "task_id": traj.get("task_id"),
                "domain": traj.get("domain"),
                "step": a.get("step"),
            }
        })
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Build per-tool SFT JSONL from unified trajectories.")
    ap.add_argument("--unified_glob", default="data/unified/*/*.json")
    ap.add_argument("--templates", default="configs/sft_templates.yaml")
    ap.add_argument("--out_dir", default="data/sft")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    args = ap.parse_args()

    random.seed(args.seed)
    tpls = load_templates(args.templates)

    files = sorted(glob.glob(args.unified_glob))
    buckets: Dict[str, List[Dict[str, Any]]] = {}

    for fp in files:
        traj = read_traj(fp)
        for ex in to_examples(traj, tpls):
            buckets.setdefault(ex["tool"], []).append(ex)

    # write per-tool train/val
    summary = {}
    for tool, rows in buckets.items():
        random.shuffle(rows)
        n = len(rows)
        k = max(1, int(n * args.val_ratio)) if n > 1 else 1
        val = rows[:k]
        trn = rows[k:] if n > 1 else rows[:]
        write_jsonl(os.path.join(args.out_dir, tool, "train.jsonl"), trn)
        write_jsonl(os.path.join(args.out_dir, tool, "val.jsonl"), val)
        summary[tool] = {"train": len(trn), "val": len(val)}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[ok] wrote per-tool SFT sets to", args.out_dir)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()