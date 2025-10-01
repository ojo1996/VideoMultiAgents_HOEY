# Multi-Agent Runner

This repository contains scripts to run all agent types end-to-end to generate real trajectories for training and evaluation.

## 🚀 Quick Start

### Option 1: Mock LLM (No API Key Required) - RECOMMENDED
```bash
# Run all agents with mock LLM responses (no API key needed)
python run_agents_with_mock.py --num_trajectories 10

# Run only specific agents
python run_agents_with_mock.py --agents mhqa math swe --num_trajectories 5

# Skip generation, only convert existing trajectories
python run_agents_with_mock.py --skip_generation
```

### Option 2: Real LLM (Requires OpenAI API Key)
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run all agents with real LLM responses
python run_all_agents.py --num_trajectories 10

# Run only specific agents
python run_all_agents.py --agents mhqa math swe --num_trajectories 5
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
- **Two Modes**: Mock LLM (no API key) or Real LLM (requires OpenAI API key)

## Agent Types

### MHQA (Multi-Hop Question Answering)
- **Tool-based**: Uses BM25 search, dense search, hybrid merge, and heuristic reader
- **LLM-based**: Uses language models for reasoning and decomposition
- **No API Key Required**: Tool-based version works without external dependencies

### Math Agent
- **Mock Mode**: Generates realistic mock trajectories for testing
- **Real Mode**: Requires OpenAI API key for actual math problem solving
- **Features**: Step-by-step reasoning, code execution, validation

### SWE (Software Engineering) Agent
- **Tool-based**: Uses bash execution and file editing tools
- **No API Key Required**: Works with actual code execution
- **Features**: Code generation, testing, debugging

### Video Agent
- **Mock Mode**: Generates mock video analysis trajectories
- **Real Mode**: Requires OpenAI API key for actual video understanding
- **Features**: Video context loading, scene analysis, object detection

### TAU (Text Analysis) Agent
- **Mock Mode**: Generates mock policy analysis trajectories
- **Real Mode**: Requires OpenAI API key for actual policy analysis
- **Features**: Context loading, policy reasoning, decision making

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
