import argparse, json, os, uuid, datetime
from typing import Dict, Any, Optional
from .Tools import BM25SearchTool, DenseSearchTool, HybridMergeTool, HeuristicReader, ToolResult

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def new_step(turn_id: int, phase: str, thought: str, action_block: Optional[str] = None,
             observation: Dict[str, Any] = None):
    content = thought
    if action_block:
        content += f"\n\n```bash\n{action_block}\n```"
    obs_xml = ""
    if observation:
        rc = observation.get("returncode", 0)
        out = observation.get("stdout", "")
        obs_xml = f"<returncode>{rc}</returncode><output>{out}</output>"
    return [
        {"role": "assistant", "content": content, "phase": phase, "turn_id": turn_id},
        {"role": "tool", "content": obs_xml, "turn_id": turn_id},
    ]

def run_episode(question: str, topk_sparse: int, topk_dense: int, out_path: str):
    run = {
        "run_id": str(uuid.uuid4()),
        "domain": "mhqa",
        "task_id": question[:128],
        "env": "local",
        "model_name": "none (stub)",
        "start_time": now_iso(),
        "end_time": None,
        "success": False,
        "stop_reason": "unknown",
        "steps": [],
    }

    bm25 = BM25SearchTool()
    dense = DenseSearchTool()
    merge = HybridMergeTool()
    reader = HeuristicReader()

    turn = 1
    run["steps"] += new_step(turn, "PLAN",
        f"PLAN: retrieve evidence via sparse and dense, merge, then read to answer: '{question}'.")

    # Sparse
    turn += 1
    r_bm25: ToolResult = bm25(question, k=topk_sparse)
    run["steps"] += new_step(turn, "RETRIEVE_SPARSE", "Retrieve with BM25.",
                             f"bm25_search(q={question!r}, k={topk_sparse})",
                             {"returncode": r_bm25.returncode, "stdout": r_bm25.stdout})

    # Dense
    turn += 1
    r_dense: ToolResult = dense(question, k=topk_dense)
    run["steps"] += new_step(turn, "RETRIEVE_DENSE", "Retrieve with dense encoder.",
                             f"dense_search(q={question!r}, k={topk_dense})",
                             {"returncode": r_dense.returncode, "stdout": r_dense.stdout})

    # Hybrid merge
    turn += 1
    r_hybrid: ToolResult = merge(r_bm25.stdout or "{}", r_dense.stdout or "{}", k=max(topk_sparse, topk_dense))
    run["steps"] += new_step(turn, "HYBRID", "Merge and dedupe.",
                             "hybrid_merge(bm25_json, dense_json)",
                             {"returncode": r_hybrid.returncode, "stdout": r_hybrid.stdout})

    # Reader
    turn += 1
    r_read: ToolResult = reader(question, r_hybrid.stdout or "{}")
    run["steps"] += new_step(turn, "READ", "Pick an answer span from merged context.",
                             "heuristic_reader(question, merged_json)",
                             {"returncode": r_read.returncode, "stdout": r_read.stdout})

    # Finalize
    turn += 1
    run["steps"] += new_step(turn, "FINALIZE", "Return the answer.", "echo SUBMIT_FINAL",
                             {"returncode": 0, "stdout": r_read.stdout})

    run["end_time"] = now_iso()
    run["success"] = (r_read.returncode == 0)
    run["stop_reason"] = "solved" if run["success"] else "error"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    print(f"[ok] saved trajectory to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="Multi-hop question text")
    ap.add_argument("--topk_sparse", type=int, default=5)
    ap.add_argument("--topk_dense", type=int, default=5)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"data/raw/{uuid.uuid4()}.traj.json"
    run_episode(args.question, args.topk_sparse, args.topk_dense, out_path)