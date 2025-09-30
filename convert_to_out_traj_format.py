#!/usr/bin/env python3
"""
convert_to_out_traj_format.py

Unifies raw trajectory JSONs from different agents into a common schema:

{
  "task_id": str,
  "domain": str,              # e.g., "swe", "math", "video", "mhqa"
  "question": str | null,
  "actions": [                # each step/tool call the agent performed
    {
      "step": int,
      "tool": str,            # e.g., "bash", "search", "write", "reason"
      "input": str | dict,
      "output": str | dict
    }
  ],
  "final_answer": str | null,
  "metadata": dict            # anything extra we keep around
}

Usage:
  Single file:
    python convert_to_out_traj_format.py path/to/raw.traj.json --domain swe

  Folder (batch):
    python convert_to_out_traj_format.py path/to/folder --domain swe --out_dir data/unified/swe
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

Json = Union[Dict[str, Any], List[Any]]

# ------------- helpers -------------
CANDIDATE_QUESTION_KEYS = [
    "task", "question", "problem", "prompt", "query", "instruction", "title"
]
CANDIDATE_FINAL_KEYS = [
    "final_answer", "final", "answer", "solution", "result", "completion"
]

def load_json(p: Path) -> Json:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj: Json, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def find_first_key(obj: Any, keys: List[str]) -> Optional[Any]:
    """Depth-first search: return the first value for any key in `keys`."""
    if isinstance(obj, dict):
        for k in obj:
            if k in keys:
                return obj[k]
        # search nested
        for v in obj.values():
            out = find_first_key(v, keys)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for it in obj:
            out = find_first_key(it, keys)
            if out is not None:
                return out
    return None

def last_text_message(obj: Any) -> Optional[str]:
    """Try to get the last assistant text content if present."""
    # works for many chat-like dumps
    if isinstance(obj, dict):
        # common: {"messages": [{"role": "assistant", "content": "..."}]}
        if "messages" in obj and isinstance(obj["messages"], list):
            for m in reversed(obj["messages"]):
                if isinstance(m, dict) and m.get("role") in {"assistant", "final"}:
                    content = m.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
    if isinstance(obj, list):
        # if list of messages/steps
        for m in reversed(obj):
            if isinstance(m, dict):
                content = m.get("content") or m.get("output")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return None

def extract_actions_generic(obj: Any) -> List[Dict[str, Any]]:
    """
    Try to pull a best-effort action list from common raw trajectories.
    Looks for 'steps', 'actions', 'tool_calls', etc. Falls back to empty list.
    """
    # 1) direct keys
    for cand in ["actions", "steps", "tool_calls", "calls"]:
        val = obj.get(cand) if isinstance(obj, dict) else None
        if isinstance(val, list) and all(isinstance(x, dict) for x in val):
            # normalize keys
            out = []
            for i, a in enumerate(val, 1):
                out.append({
                    "step": a.get("step", i),
                    "tool": a.get("tool") or a.get("name") or a.get("type") or "unknown",
                    "input": a.get("input") or a.get("args") or a.get("request") or "",
                    "output": a.get("output") or a.get("response") or ""
                })
            return out

    # 2) message-style logs that embed tool runs
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        out = []
        step = 1
        for m in obj["messages"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            # heuristic: bash/tool commands show up as 'tool'/'system' or have 'cmd'
            tool = None
            if "tool" in m: tool = m["tool"]
            elif "cmd" in m: tool = "bash"
            elif role in {"tool", "system"}: tool = "tool"
            if tool:
                out.append({
                    "step": step,
                    "tool": tool,
                    "input": m.get("cmd") or m.get("input") or m.get("content") or "",
                    "output": m.get("output") or ""
                })
                step += 1
        if out:
            return out

    return []  # we can iterate later with agent-specific parsers

def guess_source(obj: Any) -> str:
    """Lightweight source detection; extend as you add agents."""
    txt = json.dumps(obj)[:5000].lower()
    if "mini-swe-agent" in txt or "minisweagent" in txt:
        return "swe"
    if "math" in txt and "sympy" in txt:
        return "math"
    if "video" in txt:
        return "video"
    if "mhqa" in txt or "multi-hop" in txt:
        return "mhqa"
    return "unknown"

def to_unified(obj: Json, task_id: str, domain_hint: Optional[str]) -> Dict[str, Any]:
    domain = (domain_hint or guess_source(obj)).lower()

    question = find_first_key(obj, CANDIDATE_QUESTION_KEYS)
    if isinstance(question, dict):
        question = json.dumps(question)
    if question is not None:
        question = str(question)

    final_answer = find_first_key(obj, CANDIDATE_FINAL_KEYS)
    if not final_answer:
        final_answer = last_text_message(obj)

    actions = extract_actions_generic(obj)

    unified = {
        "task_id": task_id,
        "domain": domain,
        "question": question,
        "actions": actions,
        "final_answer": final_answer,
        "metadata": {
            "source_detected": guess_source(obj)
        }
    }
    return unified

def convert_path(in_path: Path, out_dir: Path, domain: Optional[str]) -> Path:
    raw = load_json(in_path)
    task_id = in_path.stem
    unified = to_unified(raw, task_id, domain)
    out_path = out_dir / f"{task_id}.json"
    dump_json(unified, out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Convert raw trajectories to unified schema.")
    ap.add_argument("input", help="Path to .json file or a directory of .json files.")
    ap.add_argument("--domain", default=None, help="Override domain label (e.g., swe, math, video, mhqa).")
    ap.add_argument("--out_dir", default="data/unified", help="Where to write unified JSON(s).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if in_path.is_dir():
        jsons = sorted([p for p in in_path.glob("*.json") if p.is_file()])
        if not jsons:
            print(f"[warn] no .json files in {in_path}", file=sys.stderr)
            sys.exit(1)
        wrote = []
        for p in jsons:
            wrote.append(convert_path(p, out_dir, args.domain))
        print(f"[ok] wrote {len(wrote)} unified files to {out_dir}")
    else:
        outp = convert_path(in_path, out_dir, args.domain)
        print(f"[ok] wrote unified file: {outp}")

if __name__ == "__main__":
    main()