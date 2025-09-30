#!/bin/bash
# Setup script for RL and SFT training pipeline

set -e

echo "=== VideoMultiAgents Training Setup ==="
echo "Setting up RL and SFT training pipeline..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw/{mhqa,video,math,swe,tau}
mkdir -p data/unified/{mhqa,video,math,swe,tau}
mkdir -p data/sft
mkdir -p data/rewards
mkdir -p models/{sft,rl}
mkdir -p results/{evaluation,plots}
mkdir -p examples

# Install training dependencies
echo "Installing training dependencies..."
pip install -r requirements_training.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py
chmod +x training/*.py
chmod +x examples/*.py

# Create sample data if it doesn't exist
if [ ! -f "data/video_samples/toy_captions.txt" ]; then
    echo "Creating sample video captions..."
    mkdir -p data/video_samples
    cat > data/video_samples/toy_captions.txt << EOF
Scene 1: A person walks into a room and sits down at a desk.
Scene 2: They open a laptop and start typing on the keyboard.
Scene 3: The person looks at the screen and smiles.
Scene 4: They close the laptop and stand up from the desk.
Scene 5: The person walks out of the room.
EOF
fi

# Test the setup
echo "Testing setup..."
python -c "
import sys
sys.path.append('.')
from scripts.calculate_rewards import RewardCalculator
from training.trl_train_rl import create_rl_dataset
print('✓ All imports successful')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Run example: python examples/train_example.py"
echo "2. Train models: python scripts/train_models.py --mode both"
echo "3. Read documentation: cat TRAINING_README.md"
echo ""
echo "Available commands:"
echo "  - Generate trajectories: python agent_systems/*/Main.py --help"
echo "  - Calculate rewards: python scripts/calculate_rewards.py --help"
echo "  - Train SFT: python training/trl_train_tool_sft.py --help"
echo "  - Train RL: python training/trl_train_rl.py --help"
echo "  - Full pipeline: python scripts/train_models.py --help"
echo ""
echo "Configuration files:"
echo "  - Main config: configs/training_config.yaml"
echo "  - SFT config: configs/trl_defaults.yaml"
echo "  - RL config: configs/rl_defaults.yaml"
echo ""
echo "Happy training! 🚀"
