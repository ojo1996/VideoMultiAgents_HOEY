import argparse, json, os, uuid, datetime
from typing import Dict, Any
from Tools import FrameSampler, TemporalReasoner, ToolResult

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def new_step(turn_id: int, phase: str, thought: str, action_block: str = None, observation: Dict[str, Any] = None):
    content = thought
    if action_block:
        content += f"\n\n```bash\n{action_block}\n```"
    obs_xml = ""
    if observation:
        rc = observation.get("returncode")
        out = observation.get("stdout", "")
        obs_xml = f"<returncode>{rc}</returncode><output>{out}</output>"
    return [
        {"role": "assistant", "content": content, "phase": phase, "turn_id": turn_id},
        {"role": "tool", "content": obs_xml, "turn_id": turn_id},
    ]

def run_episode(video_path: str, question: str, out_path: str):
    run = {
        "run_id": str(uuid.uuid4()),
        "domain": "video",
        "task_id": os.path.basename(video_path),
        "env": "local",
        "model_name": "none (stub)",
        "start_time": now_iso(),
        "end_time": None,
        "success": False,
        "stop_reason": "unknown",
        "steps": [],
    }

    sampler = FrameSampler(num_frames=8)
    reasoner = TemporalReasoner()

    turn = 1
    run["steps"] += new_step(turn, "PLAN", f"PLAN: sample frames, then answer: '{question}'.")

    turn += 1
    res1: ToolResult = sampler(video_path)
    run["steps"] += new_step(
        turn, "EXEC",
        "EXEC: sample frames uniformly.",
        f"frames = FrameSampler(num_frames=8)('{video_path}')",
        {"returncode": res1.returncode, "stdout": res1.stdout},
    )

    turn += 1
    res3: ToolResult = reasoner(question, {"frames": res1.stdout})
    run["steps"] += new_step(
        turn, "FINALIZE",
        "FINALIZE: aggregate evidence and provide answer.",
        "final = TemporalReasoner()(question, evidence)",
        {"returncode": res3.returncode, "stdout": res3.stdout},
    )

    run["end_time"] = now_iso()
    run["success"] = (res1.returncode == 0  and res3.returncode == 0)
    run["stop_reason"] = "solved" if run["success"] else "error"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    print(f"[ok] saved trajectory to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"data/raw/{uuid.uuid4()}.traj.json"
    print("video:", args.video)
    run_episode(args.video, args.question, out_path)
