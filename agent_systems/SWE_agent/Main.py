import argparse, json, os, uuid, datetime
from typing import Dict, Any, Optional
from .Tools import BashTool, FileEditTool, ToolResult

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def new_step(turn_id: int, phase: str, thought: str, action_block: Optional[str] = None, observation: Dict[str, Any] = None):
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

def run_episode(task: str, out_path: str):
    run = {
        "run_id": str(uuid.uuid4()),
        "domain": "swe",
        "task_id": task[:64],
        "env": "local",
        "model_name": "none (stub)",
        "start_time": now_iso(),
        "end_time": None,
        "success": False,
        "stop_reason": "unknown",
        "steps": [],
    }

    bash = BashTool()
    edit = FileEditTool()

    turn = 1
    run["steps"] += new_step(turn, "PLAN", f"PLAN: create a python file to solve '{task}', then execute it.")

    # WRITE
    turn += 1
    filename = "solution.py"
    program = f"print({task})" if task.isdigit() else "print('hello from swe agent')"
    res_w: ToolResult = edit.write(filename, program)
    run["steps"] += new_step(
        turn, "WRITE",
        f"WRITE: create {filename} with the intended behavior.",
        f"cat > {filename} << 'PY'\n{program}\nPY",
        {"returncode": res_w.returncode, "stdout": res_w.stdout},
    )

    # EXEC
    turn += 1
    cmd_run = f"python3 {filename}"
    res_exec: ToolResult = bash(cmd_run)
    run["steps"] += new_step(
        turn, "EXEC",
        f"EXEC: run the script {filename}.",
        cmd_run,
        {"returncode": res_exec.returncode, "stdout": res_exec.stdout},
    )

    # FINALIZE
    turn += 1
    run["steps"] += new_step(
        turn, "FINALIZE",
        "FINALIZE: done.",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        {"returncode": 0, "stdout": "submitted"},
    )

    run["end_time"] = now_iso()
    run["success"] = (res_w.returncode == 0 and res_exec.returncode == 0)
    run["stop_reason"] = "solved" if run["success"] else "error"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    print(f"[ok] saved trajectory to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g., '2+2' or a short description")
    ap.add_argument("-o", "--out", default=None, help="output raw trajectory path")
    args = ap.parse_args()
    out_path = args.out or f"data/raw/{uuid.uuid4()}.traj.json"
    run_episode(args.task, out_path)