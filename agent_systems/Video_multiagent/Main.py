# agent_systems/Video_multiagent/Main.py
from __future__ import annotations

import argparse
import os
import uuid

from .Tools import (
    LLMConfig,
    call_llm,
    new_trajectory,
    record_action,
    save_trajectory,
    load_video_context,
    VIDEO_REASON_SYSTEM,
    VIDEO_FINALIZE_SYSTEM,
)

def main():
    ap = argparse.ArgumentParser(description="Video QA agent (LLM-only) over textual video context.")
    ap.add_argument("--video_ctx", required=True, help="Path to transcript/captions JSON/TXT.")
    ap.add_argument("--question", "-q", required=True, help="Question about the video.")
    ap.add_argument("--task_id", default=None)
    ap.add_argument("--out", "-o", default="runs/video_llm.traj.json")
    args = ap.parse_args()

    cfg = LLMConfig()
    task_id = args.task_id or f"video-{uuid.uuid4().hex[:8]}"
    traj = new_trajectory(task_id=task_id, domain="video")
    traj["question"] = args.question

    # 1) Load context
    ctx = load_video_context(args.video_ctx)
    record_action(traj, "load_context", {"path": args.video_ctx}, f"{len(ctx)} chars")

    # 2) Reason
    user = (
        f"Question:\n{args.question}\n\n"
        f"Video context (textual):\n---\n{ctx}\n---\n"
        "Answer concisely. If needed, cite short evidence in parentheses."
    )
    notes = call_llm(VIDEO_REASON_SYSTEM, user, cfg).strip()
    record_action(traj, "reason", {"prompt": user}, notes)

    # 3) Finalize
    fin_user = (
        f"Question:\n{args.question}\n\n"
        f"Preliminary notes:\n{notes}\n\n"
        "Write the final answer."
    )
    final = call_llm(VIDEO_FINALIZE_SYSTEM, fin_user, cfg).strip()
    record_action(traj, "finalize", {"prompt": fin_user}, final)
    traj["final_answer"] = final

    out_path = os.path.expanduser(os.path.expandvars(args.out))
    save_trajectory(traj, out_path)
    print(f"[ok] saved trajectory -> {out_path}")

if __name__ == "__main__":
    main()