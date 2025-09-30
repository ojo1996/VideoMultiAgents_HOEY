# Training Pipeline for VideoMultiAgents

This document describes how to perform **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning (RL)** training on agent trajectories.

## Overview

The training pipeline consists of several stages:

1. **Trajectory Generation** - Generate trajectories using agent systems
2. **Data Conversion** - Convert raw trajectories to unified format
3. **Reward Calculation** - Calculate rewards for RL training
4. **SFT Training** - Train models on trajectory data
5. **RL Training** - Train models using PPO on high-reward trajectories
6. **Evaluation** - Evaluate trained models

## Quick Start

### 1. Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install training-specific requirements
pip install -r requirements_training.txt
```

### 2. Run Example

```bash
# Run a complete example
python examples/train_example.py
```

### 3. Use Comprehensive Training Script

```bash
# Train both SFT and RL models
python scripts/train_models.py --mode both

# Train only SFT models
python scripts/train_models.py --mode sft

# Train only RL models
python scripts/train_models.py --mode rl
```

## Detailed Usage

### Trajectory Generation

Generate trajectories using individual agent systems:

```bash
# MHQA agent
python agent_systems/MHQA_agent/Main.py \
    --question "Who wrote The Hobbit and what other works did they create?" \
    --out data/raw/mhqa_demo.traj.json

# Video agent
python agent_systems/Video_multiagent/Main.py \
    --video_ctx data/video_samples/toy_captions.txt \
    --question "What happens in the first scene?" \
    --out data/raw/video_demo.traj.json

# Math agent
python agent_systems/Math_agent/Main.py \
    --question "Solve the equation 2x + 5 = 13" \
    --out data/raw/math_demo.traj.json
```

### Data Conversion

Convert raw trajectories to unified format:

```bash
python convert_to_out_traj_format.py \
    data/raw/mhqa_demo.traj.json \
    --domain mhqa \
    --out data/unified/mhqa/demo.json
```

### Reward Calculation

Calculate rewards for trajectories:

```bash
python scripts/calculate_rewards.py \
    --trajectory_glob "data/raw/*/*.traj.json" \
    --output data/rewards/analysis.json \
    --stats
```

### SFT Training

Build SFT datasets and train models:

```bash
# Build datasets
python scripts/build_sft_datasets.py \
    --unified_glob "data/unified/*/*.json" \
    --templates configs/sft_templates.yaml \
    --out_dir data/sft

# Train SFT models
python training/trl_train_tool_sft.py \
    --config configs/trl_defaults.yaml \
    --model_name Qwen/Qwen2.5-0.5B \
    --tool reason \
    --data_root data/sft \
    --out_dir models/sft
```

### RL Training

Train RL models using PPO:

```bash
python training/trl_train_rl.py \
    --config configs/rl_defaults.yaml \
    --model_name Qwen/Qwen2.5-0.5B \
    --trajectory_glob "data/raw/*/*.traj.json" \
    --out_dir models/rl \
    --min_reward 0.3
```

## Configuration

### Training Configuration

Edit `configs/training_config.yaml` to customize:

- **Agents**: Which agents to use and their questions
- **Models**: Model names and output directories
- **Rewards**: Reward calculation parameters
- **Training**: SFT and RL training parameters

### SFT Configuration

Edit `configs/trl_defaults.yaml` for SFT training:

```yaml
model_name: Qwen/Qwen2.5-0.5B
learning_rate: 2.0e-5
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
```

### RL Configuration

Edit `configs/rl_defaults.yaml` for RL training:

```yaml
model_name: Qwen/Qwen2.5-0.5B
learning_rate: 1.0e-5
batch_size: 4
ppo_epochs: 4
min_reward: 0.0
```

## Reward System

The reward system evaluates trajectories on multiple dimensions:

### Reward Components

1. **Completion** (0.4 weight) - Task completion success
2. **Efficiency** (0.2 weight) - Tool usage efficiency
3. **Quality** (0.3 weight) - Answer quality (domain-specific)
4. **Reasoning** (0.1 weight) - Reasoning step quality

### Domain-Specific Quality

- **MHQA**: Multi-hop reasoning indicators, evidence citations
- **Video**: Video-specific content, temporal references
- **Math**: Mathematical reasoning, numerical content
- **SWE**: Code quality, best practices
- **TAU**: Policy awareness, tool appropriateness

### Example Reward Calculation

```python
from scripts.calculate_rewards import RewardCalculator

calculator = RewardCalculator()
rewards = calculator.calculate_reward(trajectory, domain='mhqa')
print(f"Total reward: {rewards['total']:.3f}")
print(f"Completion: {rewards['completion']:.3f}")
print(f"Quality: {rewards['quality']:.3f}")
```

## File Structure

```
data/
├── raw/                    # Raw trajectory files
│   ├── mhqa/
│   ├── video/
│   ├── math/
│   ├── swe/
│   └── tau/
├── unified/               # Converted trajectories
│   ├── mhqa/
│   ├── video/
│   ├── math/
│   ├── swe/
│   └── tau/
├── sft/                   # SFT datasets
│   ├── reason/
│   ├── bash/
│   └── finalize/
└── rewards/               # Reward analysis
    └── analysis.json

models/
├── sft/                   # SFT models
│   ├── reason/
│   ├── bash/
│   └── finalize/
└── rl/                    # RL models

configs/
├── training_config.yaml   # Main training config
├── trl_defaults.yaml     # SFT config
└── rl_defaults.yaml      # RL config
```

## Advanced Usage

### Custom Reward Functions

Create custom reward functions by extending `RewardCalculator`:

```python
class CustomRewardCalculator(RewardCalculator):
    def _calculate_quality_reward(self, trajectory, domain):
        # Your custom quality evaluation
        return custom_score
```

### Custom Training Loops

Implement custom training logic:

```python
from training.trl_train_rl import create_rl_dataset

# Create custom dataset
dataset = create_rl_dataset(trajectory_paths, min_reward=0.5)

# Custom training loop
for epoch in range(num_epochs):
    for batch in dataset:
        # Your training logic
        pass
```

### Model Evaluation

Evaluate trained models:

```bash
python eval.py \
    --model_path models/sft/reason \
    --test_data data/test/reason.jsonl
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Low Rewards**: Adjust reward weights or improve trajectory quality
3. **Training Instability**: Lower learning rate or adjust PPO parameters
4. **Data Issues**: Check trajectory format and reward calculation

### Debug Mode

Enable debug logging:

```bash
export TRANSFORMERS_VERBOSITY=debug
python scripts/train_models.py --mode sft
```

### Monitoring

Use TensorBoard for training monitoring:

```bash
tensorboard --logdir models/sft/reason/logs
```

## Performance Tips

1. **Use GPU**: Set `CUDA_VISIBLE_DEVICES` for GPU training
2. **Batch Processing**: Process multiple trajectories in parallel
3. **Data Caching**: Cache processed datasets for faster iteration
4. **Model Checkpointing**: Save checkpoints regularly
5. **Reward Filtering**: Filter low-reward trajectories early

## Examples

See `examples/train_example.py` for a complete working example.

## Contributing

When adding new features:

1. Update reward calculation functions
2. Add new agent types to training config
3. Implement domain-specific quality evaluation
4. Add tests for new functionality
5. Update documentation

## License

Same as the main project license.
