import argparse, json, os, uuid, datetime, re
from typing import Dict, Any, Optional
from Tools import AddTool, MultiplyTool, SquareTool, eval_expression, ToolResult

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def new_step(turn_id: int, phase: str, thought: str, action_block: Optional[str] = None,
             observation: Dict[str, Any] = None):
    content = thought
    if action_block:
        content += f"\n\n```bash\n{action_block}\n```"
    obs_xml = ""
    if observation:
        rc = observation.get("returncode"); out = observation.get("stdout","")
        obs_xml = f"<returncode>{rc}</returncode><output>{out}</output>"
    return [
        {"role": "assistant", "content": content, "phase": phase, "turn_id": turn_id},
        {"role": "tool", "content": obs_xml, "turn_id": turn_id},
    ]

def pick_and_run_tool(query: str) -> (str, ToolResult):
    add = re.match(r"(?i)\s*add\s+([\-0-9\.]+)\s+(and|&)\s+([\-0-9\.]+)", query)
    mul = re.match(r"(?i)\s*(multiply|times)\s+([\-0-9\.]+)\s+(by|x)\s+([\-0-9\.]+)", query)
    sqr = re.match(r"(?i)\s*square\s+([\-0-9\.]+)", query)
    if add:
        a, b = add.group(1), add.group(3)
        return "add", AddTool()(a, b)
    if mul:
        a, b = mul.group(2), mul.group(4)
        return "multiply", MultiplyTool()(a, b)
    if sqr:
        x = sqr.group(1)
        return "square", SquareTool()(x)
    return "eval_expression", eval_expression(query)

def run_episode(task: str, out_path: str):
    run = {
        "run_id": str(uuid.uuid4()),
        "domain": "math",
        "task_id": task[:64],
        "env": "local",
        "model_name": "none (stub)",
        "start_time": now_iso(),
        "end_time": None,
        "success": False,
        "stop_reason": "unknown",
        "steps": [],
    }

    turn = 1
    run["steps"] += new_step(turn, "PLAN", f"PLAN: decide the math operation for: '{task}'.")

    turn += 1
    tool_name, res = pick_and_run_tool(task)
    action_cmd = f"{tool_name}('{task}')"
    run["steps"] += new_step(
        turn, "EXEC",
        f"EXEC: call tool {tool_name} to compute the result.",
        action_cmd,
        {"returncode": res.returncode, "stdout": res.stdout},
    )

    turn += 1
    run["steps"] += new_step(turn, "FINALIZE", "FINALIZE: return the numeric result.", "echo SUBMIT_FINAL",
                             {"returncode": 0, "stdout": res.stdout})

    run["end_time"] = now_iso()
    run["success"] = (res.returncode == 0)
    run["stop_reason"] = "solved" if run["success"] else "error"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    print(f"[ok] saved trajectory to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help='e.g., "add 3 and 5", "multiply 2 by 7", "square 9", or "2+3*4"')
    ap.add_argument("-o", "--out", default=None, help="output raw trajectory path")
    args = ap.parse_args()
    out_path = args.out or f"data/raw/{uuid.uuid4()}.traj.json"
    run_episode(args.task, out_path)