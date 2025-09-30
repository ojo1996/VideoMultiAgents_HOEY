# SFT Training Pipeline for Multi-Agent Systems

This repository contains a comprehensive Supervised Fine-Tuning (SFT) pipeline for training specialized models on agent trajectories across multiple domains.

## 🚀 Quick Start

### SWE Agent Training
```bash
# Run the complete SWE SFT pipeline
python scripts/run_swe_sft_pipeline.py --num_trajectories 10

# Or run individual steps
python scripts/generate_swe_trajectories.py --num_samples 10
python scripts/build_swe_sft_datasets.py
python training/sft_train_swe_tool.py --tool bash --data_file data/sft/swe/swe_bash_sft.jsonl --out_dir models/swe/bash
```

### MHQA Agent Training (Already Implemented)
```bash
# Generate trajectories
python -m agent_systems.MHQA_agent.Main --question "Your question here" --out data/raw/mhqa/sample.traj.json

# Train SFT model
python training/sft_train_3b_final.py --model_name Qwen/Qwen2.5-3B --trajectory_glob "data/raw/*/*.traj.json" --out_dir models/sft_3b
```

## 📁 Project Structure

```
├── agent_systems/           # Agent implementations
│   ├── SWE_agent/          # Software Engineering agent
│   ├── MHQA_agent/         # Multi-hop QA agent
│   ├── Math_agent/         # Math problem solving agent
│   ├── Video_multiagent/   # Video understanding agent
│   └── TAU_agent/          # Text Analysis agent
├── training/               # SFT training scripts
│   ├── sft_train_3b_final.py      # General SFT training
│   ├── sft_train_swe_tool.py      # SWE tool-specific training
│   └── trl_train_rl_*.py          # RL training scripts
├── scripts/                # Pipeline orchestration
│   ├── generate_swe_trajectories.py
│   ├── build_swe_sft_datasets.py
│   └── run_swe_sft_pipeline.py
├── data/                   # Training data
│   ├── raw/               # Raw trajectory files
│   ├── sft/               # SFT training datasets
│   └── unified/           # Unified trajectory format
└── models/                # Trained models
    ├── sft/               # General SFT models
    └── swe/               # SWE tool-specific models
```

## 🔧 SWE Agent Pipeline

### 1. Trajectory Generation
The SWE agent generates trajectories by solving programming tasks:

**Tools Available:**
- `bash`: Execute shell commands
- `file_edit`: Create and modify files

**Example Trajectory Structure:**
```json
{
  "run_id": "uuid",
  "domain": "swe",
  "task_id": "Create a Python function...",
  "success": true,
  "steps": [
    {
      "role": "assistant",
      "content": "PLAN: create a python file to solve 'task'",
      "phase": "PLAN",
      "turn_id": 1
    },
    {
      "role": "assistant", 
      "content": "WRITE: create solution.py\n\n```bash\ncat > solution.py << 'PY'\nprint('hello')\nPY\n```",
      "phase": "WRITE",
      "turn_id": 2
    }
  ]
}
```

### 2. Tool-Specific Dataset Building
Trajectories are grouped by tool and converted to SFT format:

**Bash Tool Examples:**
```
User: Execute the following bash command for write phase: ls -la
Assistant: ```bash
ls -la
```
```

**File Edit Tool Examples:**
```
User: Perform file editing operation for write phase
Assistant: WRITE: create solution.py with the intended behavior.
```

### 3. Model Training
Each tool gets its own specialized model:

```bash
# Train bash command model
python training/sft_train_swe_tool.py \
  --tool bash \
  --data_file data/sft/swe/swe_bash_sft.jsonl \
  --out_dir models/swe/bash

# Train file editing model  
python training/sft_train_swe_tool.py \
  --tool file_edit \
  --data_file data/sft/swe/swe_file_edit_sft.jsonl \
  --out_dir models/swe/file_edit
```

## 🧠 MHQA Agent Pipeline

### 1. Trajectory Generation
```bash
python -m agent_systems.MHQA_agent.Main \
  --question "Who wrote The Hobbit and what other works did they create?" \
  --out data/raw/mhqa/sample.traj.json
```

### 2. SFT Training
```bash
python training/sft_train_3b_final.py \
  --model_name Qwen/Qwen2.5-3B \
  --trajectory_glob "data/raw/*/*.traj.json" \
  --out_dir models/sft_3b
```

## 📊 Performance Results

### SWE Agent (Tool-Specific Models)
- **Bash Model**: Specialized for command execution
- **File Edit Model**: Specialized for file operations
- **Training Time**: ~2-3 minutes per tool
- **Memory Usage**: Optimized for 24GB GPU

### MHQA Agent (General Model)
- **Model Size**: 3B parameters (Qwen2.5-3B)
- **Training Loss**: 2.78 (6 examples, 1 epoch)
- **Response Quality**: Coherent answers vs garbled 0.5B output
- **Memory Optimizations**: bfloat16, gradient checkpointing, 64-token sequences

## 🛠️ Technical Details

### Memory Optimizations
- **bfloat16 precision**: Reduces memory usage by ~50%
- **Gradient checkpointing**: Trades compute for memory
- **Small batch sizes**: 1-2 with gradient accumulation
- **Sequence length limits**: 64-256 tokens
- **Device mapping**: Automatic GPU memory distribution

### Training Configuration
```yaml
# Key parameters for 3B models
batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5e-6
max_length: 64
num_epochs: 1
bf16: true
gradient_checkpointing: true
```

## 🚀 Usage Examples

### Generate and Train SWE Models
```bash
# Complete pipeline
python scripts/run_swe_sft_pipeline.py --num_trajectories 15

# Individual steps
python scripts/generate_swe_trajectories.py --num_samples 15
python scripts/build_swe_sft_datasets.py
python training/sft_train_swe_tool.py --tool bash --data_file data/sft/swe/swe_bash_sft.jsonl --out_dir models/swe/bash
```

### Test Trained Models
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load trained model
model = AutoModelForCausalLM.from_pretrained("models/swe/bash")
tokenizer = AutoTokenizer.from_pretrained("models/swe/bash")

# Test inference
prompt = "Execute the following bash command for write phase: ls -la"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🔄 Extending to Other Agents

### Math Agent
```bash
# Generate trajectories
python -m agent_systems.Math_agent.Main --problem "Solve x^2 + 5x + 6 = 0" --out data/raw/math/sample.traj.json

# Train model
python training/sft_train_3b_final.py --trajectory_glob "data/raw/math/*.traj.json" --out_dir models/sft_math
```

### Video Agent
```bash
# Generate trajectories  
python -m agent_systems.Video_multiagent.Main --video_path "video.mp4" --out data/raw/video/sample.traj.json

# Train model
python training/sft_train_3b_final.py --trajectory_glob "data/raw/video/*.traj.json" --out_dir models/sft_video
```

## 📈 Monitoring and Evaluation

### Training Metrics
- **Loss curves**: Monitor convergence
- **Memory usage**: Track GPU utilization
- **Training time**: Per epoch and total
- **Sample efficiency**: Examples per epoch

### Model Evaluation
- **Response quality**: Coherence and correctness
- **Tool usage**: Proper command generation
- **Domain knowledge**: Task-specific understanding

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size and sequence length
python training/sft_train_3b_final.py --batch_size 1 --max_length 64
```

**Authentication Errors**
```bash
# Set up GitHub Personal Access Token
git config --global user.email "your@email.com"
git config --global user.name "Your Name"
```

**Import Errors**
```bash
# Install dependencies
pip install -r requirements_training.txt
```

## 📝 Contributing

1. **Add new agents**: Implement in `agent_systems/`
2. **Extend training**: Add new training scripts in `training/`
3. **Improve datasets**: Enhance data processing in `scripts/`
4. **Documentation**: Update this README

## 📄 License

This project is part of the VideoMultiAgents_HOEY research initiative.

---

**Last Updated**: September 2024  
**Status**: Active Development  
**Maintainer**: Heidy Hernandez
