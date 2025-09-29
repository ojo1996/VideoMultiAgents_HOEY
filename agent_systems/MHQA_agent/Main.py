# agent_systems/MHQA_agent/Main.py
from __future__ import annotations

import argparse
import os
import uuid
from typing import List

from .Tools import (
    LLMConfig,
    call_llm,
    new_trajectory,
    record_action,
    save_trajectory,
    DECOMPOSE_SYSTEM,
    HOP_REASON_SYSTEM,
    FINALIZE_SYSTEM,
)

def parse_numbered_list(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hops: List[str] = []
    for ln in lines:
        # Accept "1. foo", "1) foo", "- foo"
        if ln[:2].isdigit() or ln[:1].isdigit() or ln.startswith(("-", "*")) or ln[:2] in {"1.", "1)"}:
            # remove leading markers
            cleaned = ln.lstrip("-* ").split(" ", 1)
            if len(cleaned) == 2 and cleaned[0].rstrip(".):").isdigit():
                hops.append(cleaned[1].strip())
            else:
                # fallback: strip common "1. " pattern
                hops.append(ln.lstrip("-* ").lstrip("0123456789.) ").strip())
        else:
            # tolerate plain lines as hops if list markers missing
            hops.append(ln)
    # prune empties and cap to 5
    return [h for h in hops if h][:5]

def main():
    ap = argparse.ArgumentParser(description="MHQA agent (LLM-only) with unified trajectory logging.")
    ap.add_argument("--question", "-q", required=True, help="Multi-hop question.")
    ap.add_argument("--task_id", default=None)
    ap.add_argument("--out", "-o", default="runs/mhqa_llm.traj.json")
    ap.add_argument("--max_hops", type=int, default=4)
    args = ap.parse_args()

    cfg = LLMConfig()
    task_id = args.task_id or f"mhqa-{uuid.uuid4().hex[:8]}"
    traj = new_trajectory(task_id=task_id, domain="mhqa")
    traj["question"] = args.question

    # 1) Decompose
    decompose_user = f"Question:\n{args.question}\n\nReturn 2–5 sub-questions as a numbered list."
    decomp = call_llm(DECOMPOSE_SYSTEM, decompose_user, cfg)
    record_action(traj, "decompose", {"prompt": decompose_user}, decomp)
    hops = parse_numbered_list(decomp)[: max(2, min(args.max_hops, 5))]

    # 2) Reason per hop
    notes: List[str] = []
    for i, hop in enumerate(hops, 1):
        user = (
            f"Original question:\n{args.question}\n\n"
            f"Prior hop notes:\n{os.linesep.join(f'- {n}' for n in notes) if notes else '(none)'}\n\n"
            f"Current sub-question ({i}/{len(hops)}): {hop}"
        )
        note = call_llm(HOP_REASON_SYSTEM, user, cfg)
        notes.append(note.strip())
        record_action(traj, "reason_hop", {"hop": hop, "idx": i, "prompt": user}, note)

    # 3) Finalize
    synth_user = (
        f"Original question:\n{args.question}\n\n"
        f"Hop notes in order:\n{os.linesep.join(f'{i+1}. {n}' for i, n in enumerate(notes))}\n\n"
        "Write the final answer."
    )
    final = call_llm(FINALIZE_SYSTEM, synth_user, cfg).strip()
    record_action(traj, "finalize", {"prompt": synth_user}, final)
    traj["final_answer"] = final

    out_path = os.path.expanduser(os.path.expandvars(args.out))
    save_trajectory(traj, out_path)
    print(f"[ok] saved trajectory -> {out_path}")

if __name__ == "__main__":
    main()