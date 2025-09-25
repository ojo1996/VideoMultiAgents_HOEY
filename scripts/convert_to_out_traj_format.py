import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable


def iter_messages_from_repo_traj(traj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # Our repo trajectories are a list of step pairs: assistant thought + tool obs
    # We'll convert to a simple chat-like format with minimal loss.
    steps = traj.get("steps", [])
    for msg in steps:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            yield {"role": "assistant", "content": content}
        elif role == "tool":
            # collapse tool observation as assistant content with a tag
            yield {"role": "assistant", "content": f"<tool_observation>{content}</tool_observation>"}


def to_unified_record(traj: Dict[str, Any]) -> Dict[str, Any]:
    messages = list(iter_messages_from_repo_traj(traj))
    meta = {
        "task_id": traj.get("task_id", traj.get("run_id", "unknown")),
        "domain": traj.get("domain", "unknown"),
        "tool_used": traj.get("used_tools", []),
        "success": bool(traj.get("success", False)),
    }
    return {"messages": messages, "meta": meta}


def sft_view(record: Dict[str, Any]) -> Dict[str, Any]:
    # Prune to role/content; add lightweight tool tag if the assistant used tools
    sft_msgs = []
    for m in record["messages"]:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant" and "<tool_observation>" in content:
            content = "<tool=observation>" + content
        sft_msgs.append({"role": role, "content": content})
    return {"messages": sft_msgs, "meta": record.get("meta", {})}


def main():
    ap = argparse.ArgumentParser(description="Convert repo trajectories to unified JSONL schema for SFT/RL")
    ap.add_argument("--in_traj", required=True, help="Input trajectory JSON (our repo format)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (unified schema)")
    ap.add_argument("--sft_only", action="store_true", help="Write SFT-pruned view instead of full record")
    args = ap.parse_args()

    with open(args.in_traj, "r", encoding="utf-8") as f:
        traj = json.load(f)

    record = to_unified_record(traj)
    if args.sft_only:
        record = sft_view(record)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()


