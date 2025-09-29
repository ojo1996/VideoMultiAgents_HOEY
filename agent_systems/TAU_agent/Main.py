# agent_systems/TAU_agent/Main.py
from __future__ import annotations

import argparse, os, uuid
from .Tools import (
    LLMConfig, call_llm, read_file,
    new_trajectory, record_action, save_trajectory,
    TAU_SYSTEM_PROMPT, FINALIZE_SYSTEM_PROMPT
)

def load_context_if_any(path, traj):
    if not path: return ""
    try:
        txt = read_file(path)
        record_action(traj, "load_context", {"path": path}, f"{len(txt)} chars")
        return txt
    except Exception as e:
        record_action(traj, "load_context", {"path": path}, f"ERROR: {e}")
        return ""

def main():
    ap = argparse.ArgumentParser(description="TAU agent: policy/tool-aware Q&A.")
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--context", "-c", default=None)
    ap.add_argument("--subdomain", choices=["retail", "airline"], default="retail")
    ap.add_argument("--task_id", default=None)
    ap.add_argument("--out", "-o", default="runs/tau_agent.traj.json")
    args = ap.parse_args()

    cfg = LLMConfig()
    task_id = args.task_id or f"tau-{uuid.uuid4().hex[:8]}"

    traj = new_trajectory(task_id=task_id, domain="tau")
    traj["question"] = args.question
    traj["metadata"]["subdomain"] = args.subdomain

    context = load_context_if_any(args.context, traj)
    user = f"Question:\n{args.question}\n\nSubdomain: {args.subdomain}\n\n" + \
           (f"Context:\n{context}" if context else "Context: (none)")

    preliminary = call_llm(TAU_SYSTEM_PROMPT, user, cfg=cfg)
    record_action(traj, "reason", {"prompt": user}, preliminary)

    final = call_llm(FINALIZE_SYSTEM_PROMPT, preliminary, cfg=cfg)
    record_action(traj, "finalize", {}, final)

    traj["final_answer"] = final.strip()
    save_trajectory(traj, os.path.expanduser(args.out))
    print(f"[ok] saved trajectory -> {args.out}")

if __name__ == "__main__":
    main()