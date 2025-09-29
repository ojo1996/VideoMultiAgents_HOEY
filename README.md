# ReasoningVectors – Merge + Eval Pipeline

This repo lets you merge RL/SFT task vectors into a base model (e.g. Qwen2.5-3B) and evaluate the merged models on **AIME25** using `lm-evaluation-harness`.

---

```powershell
# setup a venv and activate
python -m venv .venv
.\.venv\Scripts\Activate

# install dependencies
pip install -r requirements.txt

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .

# download models 
mkdir models

huggingface-cli download Qwen/Qwen2.5-3B `
  --local-dir models\AFM-MHQA-Agent-3B-sft `
  --local-dir-use-symlinks False

huggingface-cli download PersonalAILab/AFM-MHQA-Agent-3B-rl `
  --local-dir models\AFM-MHQA-Agent-3B-rl `
  --local-dir-use-symlinks False
huggingface-cli download PersonalAILab/AFM-MHQA-Agent-3B-sft `
  --local-dir models\AFM-MHQA-Agent-3B-sft `
  --local-dir-use-symlinks False

# Generate the resoning vector and apply it with various alphas
python generate_merged_models.py `
  --base_model Qwen/Qwen2.5-3B `
  --rl_model   models/AFM-MHQA-Agent-3B-rl `
  --sft_model  models/AFM-MHQA-Agent-3B-sft `
  --out_root   merges

# evaluate the models saving the results and samples.
python eval.py

# Agent Systems – Unified Trajectories & AFM Prep

This repo hosts lightweight **domain agents** (SWE, Math, Video, MHQA, TAU) that all emit a **standardized trajectory JSON** (`reason → tool calls → finalize`) to enable analysis, AFM-style distillation, and task/reasoning vector extraction.

## What’s here
- **Agents**
  - `agent_systems/SWE_agent`: shell-oriented SWE actions with Windows/Unix handling
  - `agent_systems/Math_agent`: safe `python -c "print(<expr>)"` execution
  - `agent_systems/MHQA_agent`: multi-hop QA (decompose → reason per hop → finalize)
  - `agent_systems/Video_multiagent`: QA over textual video context (captions/transcripts)
  - `agent_systems/TAU_agent`: policy-aware, τ-bench–style Q&A over provided rules/context

- **Trajectories**: written to `runs/*.traj.json` (one per run)
- **Converter**: `convert_to_out_traj_format.py` → `data/unified/<domain>/...`
- **(Optional) Validator**: `tools/validate_traj.py` + `tools/trajectory_schema.json`
- **Samples**: `data/video_samples/toy_captions.txt` (toy context); more to come

> All agents use OpenAIs Python SDK. Set `OPENAI_API_KEY` and (optionally) `OPENAI_MODEL` (defaults to `gpt-4o-mini`).

---

## Quickstart

### 1) Environment
```bash
# Create and activate venv
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Set your key (one-time per machine or per session)
# Windows (persist):
setx OPENAI_API_KEY "sk-..."
# macOS/Linux (session):
export OPENAI_API_KEY="sk-..."

Unified Trajectory Format (summary)

{
  "task_id": "domain-<id>",
  "domain": "swe|math|video|mhqa|tau",
  "question": "...",
  "actions": [
    {"step": 1, "tool": "reason", "...": "..."},
    {"step": 2, "tool": "bash|load_context|reason_hop|...", "...": "..."},
    {"step": 3, "tool": "finalize", "...": "..."}
  ],
  "final_answer": "...",
  "metadata": {"start_ts": 0.0, "end_ts": 0.0, "...": "..."}
}

Running each agent:

SWE (Software Engineering)

Cross-platform shell with Windows/Unix awareness.

python -m agent_systems.SWE_agent.Main \
  --question "List files in the repo root and count .py files" \
  --steps 1 --yolo \
  -o runs/swe_llm.traj.json

Output: reason → bash → finalize, with metadata.executed_cmds recorded.

Math

Executes exactly one safe command: python -c "print(<expr>)".

python -m agent_systems.Math_agent.Main \
  --question "23*19 + 7**3" \
  -o runs/math_llm.traj.json

MHQA (Multi-Hop QA)

Decompose → reason per hop → finalize (LLM-only; retrieval optional later).

python -m agent_systems.MHQA_agent.Main \
  --question "Who founded the company that owns Instagram?" \
  -o runs/mhqa_llm.traj.json

Video QA (textual context)

Works over transcripts/captions; no heavy video deps.

python -m agent_systems.Video_multiagent.Main \
  --video_ctx data/video_samples/toy_captions.txt \
  --question "What did the person take from the fridge and what did they do with it?" \
  -o runs/video_llm.traj.json

TAU (τ-bench–style policy QA)

Reads optional policy/task context and answers under those rules.

python -m agent_systems.TAU_agent.Main \
  --question "Customer wants to exchange two delivered items; what must I confirm first?" \
  --subdomain retail \
  --context data/tau_samples/retail_policy_excerpt.md \
  -o runs/tau_llm.traj.json

Convert trajectories to unified dataset

# Per domain (edit globs as needed)
python convert_to_out_traj_format.py "runs/swe_*.traj.json"   --domain swe   --out_dir data/unified/swe
python convert_to_out_traj_format.py "runs/math_*.traj.json"  --domain math  --out_dir data/unified/math
python convert_to_out_traj_format.py "runs/video_*.traj.json" --domain video --out_dir data/unified/video
python convert_to_out_traj_format.py "runs/mhqa_*.traj.json"  --domain mhqa  --out_dir data/unified/mhqa
python convert_to_out_traj_format.py "runs/tau_*.traj.json"   --domain tau   --out_dir data/unified/tau

Repo structure (key paths)

agent_systems/
  SWE_agent/      Main.py  Tools.py  __init__.py
  Math_agent/     Main.py  Tools.py  __init__.py
  MHQA_agent/     Main.py  Tools.py  __init__.py
  Video_multiagent/ Main.py Tools.py __init__.py
  TAU_agent/      Main.py  Tools.py  __init__.py
convert_to_out_traj_format.py
runs/                 # raw trajectories from local runs
data/
  unified/<domain>/   # converted outputs (per domain)
  video_samples/      # toy input for video
  tau_samples/        # policy/task snippets (optional)
tools/
  trajectory_schema.json       # optional
  validate_traj.py             # optional
