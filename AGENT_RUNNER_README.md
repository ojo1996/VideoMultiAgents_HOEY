# Multi-Agent Runner

This script runs all agent types end-to-end to generate real trajectories for training and evaluation.

## Quick Start

### Run All Agents
```bash
# Run all agents with 10 trajectories each
python run_all_agents.py --num_trajectories 10

# Run only specific agents
python run_all_agents.py --agents mhqa math swe --num_trajectories 5

# Skip generation, only convert existing trajectories
python run_all_agents.py --skip_generation
```

### Run Individual Agents

**MHQA Agent (LLM-based):**
```bash
python -m agent_systems.MHQA_agent.Main \
  --question "Who wrote The Hobbit and what other famous works did they create?" \
  --out data/raw/mhqa/real_question.traj.json
```

**MHQA Agent (Tool-based):**
```bash
python -m agent_systems.MHQA.main \
  --question "What company owns Instagram and who founded that company?" \
  --traj_out data/raw/mhqa/tool_question.traj.json \
  --vectors_out data/generated_results/tool_vectors.json
```

**Math Agent:**
```bash
python -m agent_systems.Math_agent.Main \
  --question "Solve the equation 2x + 5 = 13" \
  --out data/raw/math/real_math.traj.json
```

**SWE Agent:**
```bash
python -m agent_systems.SWE_agent.Main \
  --question "Write a Python function to calculate the factorial of a number" \
  --out data/raw/swe/real_swe.traj.json
```

**Video Agent:**
```bash
python -m agent_systems.Video_multiagent.Main \
  --video_ctx data/video_samples/toy_captions.txt \
  --question "What happens in the first scene of the video?" \
  --out data/raw/video/real_video.traj.json
```

**TAU Agent:**
```bash
python -m agent_systems.TAU_agent.Main \
  --question "How should I handle a customer complaint about a delayed order?" \
  --subdomain retail \
  --context data/tau_samples/retail_policy_excerpt.md \
  --out data/raw/tau/real_tau.traj.json
```

## Features

- **Real Datasets**: Uses actual questions from config files, not just samples
- **Both Agent Types**: Runs both LLM-based and tool-based MHQA agents
- **End-to-End**: Generates trajectories → converts to unified format → ready for training
- **Comprehensive**: Covers all 5 agent types (MHQA, Math, SWE, Video, TAU)
- **Scalable**: Can generate any number of trajectories per agent
- **Robust**: Includes error handling and progress tracking

## Output Structure

```
data/
├── raw/                    # Raw trajectory files
│   ├── mhqa/              # MHQA trajectories (LLM + Tool-based)
│   ├── math/              # Math agent trajectories
│   ├── swe/               # SWE agent trajectories
│   ├── video/             # Video agent trajectories
│   └── tau/               # TAU agent trajectories
├── unified/               # Converted to unified format
│   ├── mhqa/
│   ├── math/
│   ├── swe/
│   ├── video/
│   └── tau/
└── generated_results/     # Tool vectors and metadata
```

## Training on Generated Data

After generating trajectories, train models:

```bash
# Train on all generated trajectories
python training/sft_train_3b_final.py \
  --model_name Qwen/Qwen2.5-3B \
  --trajectory_glob "data/raw/*/*.traj.json" \
  --out_dir models/sft_3b_real
```

## Troubleshooting

- **Missing dependencies**: Make sure all agent dependencies are installed
- **Permission errors**: Ensure the script is executable (`chmod +x run_all_agents.py`)
- **Timeout issues**: Increase timeout in the script if agents take longer to run
- **Missing context files**: Ensure video and TAU context files exist before running

## Configuration

The script uses questions from `configs/training_config.yaml`. You can modify the questions there or pass custom questions to individual agents.
