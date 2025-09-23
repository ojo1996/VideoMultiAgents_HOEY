import argparse, json, os, uuid, datetime
from xml.sax import saxutils as _xml_saxutils
from typing import Dict, Any, Optional

# The tool layer is deliberately small and model-agnostic.
from .tools import (
    register_default_tools,
    tool_to_vector,
    validate_tool_vector,
    text_hash_vector,
    ToolResult,
)


#
# Runner for a minimal MHQA-style agent episode.
#
# Responsibilities:
# - Build tool registry and standardized tool vectors
# - Validate tool vectors to guarantee downstream integrity
# - Produce a simple task vector
# - Execute a deterministic pipeline: bm25 -> dense -> merge -> reader
# - Record a standardized trajectory file plus sidecar vectors
#


def now_iso() -> str:
    """UTC ISO8601 with trailing Z, without microseconds."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def new_step(
    turn_id: int,
    phase: str,
    thought: str,
    action_block: Optional[str] = None,
    observation: Dict[str, Any] = None,
):
    """Create a two-message step: assistant thought + tool observation.

    Observations are rendered as a simple XML snippet to match existing
    trajectory tooling in this repository.
    """
    content = thought
    if action_block:
        content += f"\n\n```bash\n{action_block}\n```"
    obs_xml = ""
    if observation:
        rc = observation.get("returncode")
        out = observation.get("stdout", "")
        # Escape XML special characters to ensure well-formed content
        safe_out = _xml_saxutils.escape(str(out), {"\"": "&quot;", "'": "&apos;"})
        obs_xml = f"<returncode>{rc}</returncode><output>{safe_out}</output>"
    return [
        {"role": "assistant", "content": content, "phase": phase, "turn_id": turn_id},
        {"role": "tool", "content": obs_xml, "turn_id": turn_id},
    ]


def run_episode(question: str, out_path: str, vectors_path: str):
    # 1) Tools + vectors
    reg = register_default_tools()
    tool_vectors = [tool_to_vector(s) for s in reg.specs()]
    val_errors = {tv["tool_id"]: validate_tool_vector(tv) for tv in tool_vectors}
    if any(val_errors[k] for k in val_errors):
        # Fail early so bad vectors never reach the dataset layer.
        raise ValueError(f"Invalid tool vectors: {json.dumps(val_errors, indent=2)}")

    # 2) Task vector (deterministic, model-free)
    task_vec = {
        "task_id": question[:64],
        "dim": 128,
        "vector": text_hash_vector(question, 128),
        "metadata": {"domain": "mhqa", "tags": []},
    }

    # 3) Initialize run record
    run = {
        "run_id": str(uuid.uuid4()),
        "domain": "mhqa",
        "task_id": task_vec["task_id"],
        "env": "local",
        "model_name": "none (stub)",
        "start_time": now_iso(),
        "end_time": None,
        "success": False,
        "stop_reason": "unknown",
        "steps": [],
        "used_tools": [],
        "vectors": {"task_vector": task_vec, "tool_vectors": tool_vectors},
    }

    # 4) PLAN
    turn = 1
    run["steps"] += new_step(turn, "PLAN", f"PLAN: retrieve, merge, read for: '{question}'.")

    # 5) EXEC pipeline
    turn += 1
    bm25 = reg.get("bm25_search")(question, k=5)
    run["used_tools"].append("bm25_search")
    run["steps"] += new_step(
        turn,
        "EXEC",
        "EXEC: call bm25_search.",
        "bm25_search(q,k=5)",
        {"returncode": bm25.returncode, "stdout": bm25.stdout},
    )

    turn += 1
    dense = reg.get("dense_search")(question, k=5)
    run["used_tools"].append("dense_search")
    run["steps"] += new_step(
        turn,
        "EXEC",
        "EXEC: call dense_search.",
        "dense_search(q,k=5)",
        {"returncode": dense.returncode, "stdout": dense.stdout},
    )

    turn += 1
    merged = reg.get("hybrid_merge")(bm25.stdout, dense.stdout, k=10)
    run["used_tools"].append("hybrid_merge")
    run["steps"] += new_step(
        turn,
        "EXEC",
        "EXEC: merge retrieval results.",
        "hybrid_merge(bm25,dense,k=10)",
        {"returncode": merged.returncode, "stdout": merged.stdout},
    )

    turn += 1
    read = reg.get("heuristic_reader")(question, merged.stdout)
    run["used_tools"].append("heuristic_reader")
    run["steps"] += new_step(
        turn,
        "EXEC",
        "EXEC: extract answer span.",
        "heuristic_reader(q,merged)",
        {"returncode": read.returncode, "stdout": read.stdout},
    )

    # 6) FINALIZE
    turn += 1
    run["steps"] += new_step(
        turn,
        "FINALIZE",
        "FINALIZE: return answer.",
        "echo SUBMIT_FINAL",
        {"returncode": 0, "stdout": read.stdout},
    )

    run["end_time"] = now_iso()
    run["success"] = read.returncode == 0
    run["stop_reason"] = "solved" if run["success"] else "error"

    # 7) Persist artifacts
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(vectors_path), exist_ok=True)
    with open(vectors_path, "w", encoding="utf-8") as f:
        json.dump({"task_vector": task_vec, "tool_vectors": tool_vectors}, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved trajectory to {out_path}")
    print(f"[ok] saved vectors to {vectors_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--traj_out", default="data/raw/episode.traj.json")
    ap.add_argument("--vectors_out", default="data/generated_results/tool_task_vectors.json")
    args = ap.parse_args()
    run_episode(args.question, args.traj_out, args.vectors_out)


