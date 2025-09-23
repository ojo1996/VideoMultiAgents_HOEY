## MHQA Agent (Canonical)

This directory contains the canonical, model-agnostic MHQA agent implementation.

- Standardized tool and task vectors with validation
- Deterministic pipeline (bm25 → dense → merge → reader)
- Clean `ToolRegistry` for plug-and-play tools
- Minimal dependencies, suitable for dataset generation and analysis

Usage
```bash
python -m agent_systems.MHQA_agent.main \
  --question "Who wrote The Hobbit?" \
  --traj_out data/raw/mhqa_demo.traj.json \
  --vectors_out data/generated_results/tool_task_vectors.json
```

Tools
- `BM25Search`, `DenseSearch`, `HybridMerge`, `HeuristicReader`

Artifacts
- Trajectory: saved to `--traj_out`
- Vectors: `{ task_vector, tool_vectors }` saved to `--vectors_out`

Note

```bash
python -m agent_systems.MHQA_agent.main --question "..."
```


