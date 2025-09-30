# VideoMultiAgents_HOEY

This repository combines two major components:

1. **Reasoning Vectors – Merge + Eval Pipeline**  
   Merge RL/SFT task vectors into a base model (e.g., Qwen2.5-3B) and evaluate merged models on benchmarks such as **AIME25** using `lm-evaluation-harness`.

2. **Agent Systems – Unified Trajectories**  
   Lightweight domain agents (SWE, Math, Video, MHQA, TAU) that emit a standardized trajectory JSON (`reason → tool calls → finalize`) for AFM-style distillation, task/reasoning vector extraction, and plug-and-play analysis.

---

## Setup & Installation

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install external repos
```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
```

### 4. Download models
```bash
mkdir models

huggingface-cli download Qwen/Qwen2.5-3B \
  --local-dir models/AFM-MHQA-Agent-3B-sft \
  --local-dir-use-symlinks False

huggingface-cli download PersonalAILab/AFM-MHQA-Agent-3B-rl \
  --local-dir models/AFM-MHQA-Agent-3B-rl \
  --local-dir-use-symlinks False

huggingface-cli download PersonalAILab/AFM-MHQA-Agent-3B-sft \
  --local-dir models/AFM-MHQA-Agent-3B-sft \
  --local-dir-use-symlinks False
```

---

## Reasoning Vectors – Merge + Eval

### Generate reasoning vectors and apply with alphas
```bash
python generate_merged_models.py \
  --base_model Qwen/Qwen2.5-3B \
  --rl_model   models/AFM-MHQA-Agent-3B-rl \
  --sft_model  models/AFM-MHQA-Agent-3B-sft \
  --out_root   merges
```

### Evaluate models
```bash
python eval.py
```

---

## Agent Systems – Unified Trajectories & AFM Prep

### What’s here
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

> All agents use OpenAI’s Python SDK.  
> Set `OPENAI_API_KEY` and (optionally) `OPENAI_MODEL` (defaults to `gpt-4o-mini`).

---

### Quickstart

#### 1) Set environment variables
```bash
# Windows (persist)
setx OPENAI_API_KEY "sk-..."

# macOS/Linux (session)
export OPENAI_API_KEY="sk-..."
```

#### 2) Unified Trajectory Format
```json
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
```

#### 3) Running each agent

**SWE (Software Engineering)**
```bash
python -m agent_systems.SWE_agent.Main \
  --question "List files in the repo root and count .py files" \
  --steps 1 --yolo \
  -o runs/swe_llm.traj.json
```

**Math**
```bash
python -m agent_systems.Math_agent.Main \
  --question "23*19 + 7**3" \
  -o runs/math_llm.traj.json
```

**MHQA (Multi-Hop QA)**
```bash
python -m agent_systems.MHQA_agent.Main \
  --question "Who founded the company that owns Instagram?" \
  -o runs/mhqa_llm.traj.json
```

**Video QA (textual context)**
```bash
python -m agent_systems.Video_multiagent.Main \
  --video_ctx data/video_samples/toy_captions.txt \
  --question "What did the person take from the fridge and what did they do with it?" \
  -o runs/video_llm.traj.json
```

**TAU (τ-bench–style policy QA)**
```bash
python -m agent_systems.TAU_agent.Main \
  --question "Customer wants to exchange two delivered items; what must I confirm first?" \
  --subdomain retail \
  --context data/tau_samples/retail_policy_excerpt.md \
  -o runs/tau_llm.traj.json
```

#### 4) Convert trajectories to unified dataset
```bash
python convert_to_out_traj_format.py "runs/swe_*.traj.json"   --domain swe   --out_dir data/unified/swe
python convert_to_out_traj_format.py "runs/math_*.traj.json"  --domain math  --out_dir data/unified/math
python convert_to_out_traj_format.py "runs/video_*.traj.json" --domain video --out_dir data/unified/video
python convert_to_out_traj_format.py "runs/mhqa_*.traj.json"  --domain mhqa  --out_dir data/unified/mhqa
python convert_to_out_traj_format.py "runs/tau_*.traj.json"   --domain tau   --out_dir data/unified/tau
```

#### 5) Building SFT Datasets (per-tool)

**Generate train/val splits per tool:**
```bash
python scripts/build_sft_datasets.py \
  --unified_glob "data/unified/*/*.json" \
  --templates configs/sft_templates.yaml \
  --out_dir data/sft
```
**Preview the dataset:**
```bash
python scripts/preview_sft.py --root data/sft --limit 1
```
**Example output (SUMMARY.json):**
```bash
{
  "reason": { "train": 1, "val": 1 },
  "bash": { "train": 1, "val": 1 },
  "finalize": { "train": 3, "val": 1 },
  "reason_hop": { "train": 1, "val": 1 },
  "load_context": { "train": 1, "val": 1 }
}

```
#### 6) TRL Training (per tool)

**Default config:**
```bash
configs/trl_defaults.yaml
```
**Training logic:**
```bash
training/trl_train_tool_sft.py
```
**Wrapper script:**
```bash
scripts/run_sft.sh
```
**Run training:**

```bash

./scripts/run_sft.sh reason   Qwen/Qwen2.5-0.5B
./scripts/run_sft.sh bash     Qwen/Qwen2.5-0.5B
./scripts/run_sft.sh finalize Qwen/Qwen2.5-0.5B

```
**Models are saved under:**

```bash

runs/sft/<tool>/

```
---

### Repo Structure (key paths)
```
agent_systems/
  SWE_agent/          Main.py  Tools.py  __init__.py
  Math_agent/         Main.py  Tools.py  __init__.py
  MHQA_agent/         Main.py  Tools.py  __init__.py
  Video_multiagent/   Main.py  Tools.py  __init__.py
  TAU_agent/          Main.py  Tools.py  __init__.py
configs/
  trl_defaults.yaml
convert_to_out_traj_format.py
scripts/
  build_sft_datasets.py
  preview_sft.py
  run_sft.sh
training/
  trl_train_tool_sft.py
runs/                 # raw + trained outputs
data/
  unified/<domain>/   # converted unified trajectories
  video_samples/      # toy input for video
  tau_samples/        # policy/task snippets
  sft/                # per-tool train/val splits
tools/
  trajectory_schema.json
  validate_traj.py

```
