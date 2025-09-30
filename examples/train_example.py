#!/usr/bin/env python3
"""
Example script demonstrating how to train SFT and RL models on trajectories.

This script shows the complete workflow from trajectory generation to model training.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_example():
    """Run a complete example of the training pipeline."""
    
    print("=== VideoMultiAgents Training Example ===")
    print("This example demonstrates SFT and RL training on agent trajectories.\n")
    
    # Step 1: Generate some sample trajectories
    print("1. Generating sample trajectories...")
    
    # Create output directory
    os.makedirs("data/raw/mhqa", exist_ok=True)
    
    # Generate a few MHQA trajectories
    questions = [
        "Who wrote The Hobbit and what other famous works did they create?",
        "What is the capital of France and what is its population?",
        "Who was the first person to walk on the moon and when did it happen?"
    ]
    
    for i, question in enumerate(questions):
        output_file = f"data/raw/mhqa/sample_{i+1}.traj.json"
        cmd = [
            sys.executable, "agent_systems/MHQA_agent/Main.py",
            "--question", question,
            "--out", output_file
        ]
        
        print(f"  Generating: {question}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"    ✓ Saved to {output_file}")
        else:
            print(f"    ✗ Failed: {result.stderr}")
    
    # Step 2: Convert trajectories to unified format
    print("\n2. Converting trajectories to unified format...")
    
    os.makedirs("data/unified/mhqa", exist_ok=True)
    
    for i in range(len(questions)):
        input_file = f"data/raw/mhqa/sample_{i+1}.traj.json"
        output_file = f"data/unified/mhqa/sample_{i+1}.json"
        
        cmd = [
            sys.executable, "convert_to_out_traj_format.py",
            input_file,
            "--domain", "mhqa",
            "--out", output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"    ✓ Converted: {input_file} -> {output_file}")
        else:
            print(f"    ✗ Failed: {result.stderr}")
    
    # Step 3: Calculate rewards
    print("\n3. Calculating rewards...")
    
    cmd = [
        sys.executable, "scripts/calculate_rewards.py",
        "--trajectory_glob", "data/raw/mhqa/*.traj.json",
        "--output", "data/rewards/sample_analysis.json",
        "--stats"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("    ✓ Rewards calculated")
        print("    " + result.stdout.split('\n')[-2])  # Print last line of stats
    else:
        print(f"    ✗ Failed: {result.stderr}")
    
    # Step 4: Build SFT datasets
    print("\n4. Building SFT datasets...")
    
    cmd = [
        sys.executable, "scripts/build_sft_datasets.py",
        "--unified_glob", "data/unified/*/*.json",
        "--templates", "configs/sft_templates.yaml",
        "--out_dir", "data/sft"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("    ✓ SFT datasets built")
    else:
        print(f"    ✗ Failed: {result.stderr}")
    
    # Step 5: Train a simple SFT model (if we have enough data)
    print("\n5. Training SFT model...")
    
    # Check if we have enough data
    sft_files = list(Path("data/sft").glob("*/train.jsonl"))
    if not sft_files:
        print("    ⚠ No SFT data found, skipping training")
    else:
        print(f"    Found {len(sft_files)} SFT datasets")
        
        # Train on the first available tool
        tool = sft_files[0].parent.name
        print(f"    Training on tool: {tool}")
        
        cmd = [
            sys.executable, "training/trl_train_tool_sft.py",
            "--config", "configs/trl_defaults.yaml",
            "--model_name", "Qwen/Qwen2.5-0.5B",
            "--tool", tool,
            "--data_root", "data/sft",
            "--out_dir", "models/sft"
        ]
        
        print("    Note: This may take a while...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"    ✓ SFT model trained for {tool}")
        else:
            print(f"    ✗ Failed: {result.stderr}")
    
    # Step 6: Show how to train RL (without actually doing it due to complexity)
    print("\n6. RL Training (example command)...")
    print("    To train RL models, you would run:")
    print("    python training/trl_train_rl.py \\")
    print("        --config configs/rl_defaults.yaml \\")
    print("        --model_name Qwen/Qwen2.5-0.5B \\")
    print("        --trajectory_glob 'data/raw/*/*.traj.json' \\")
    print("        --out_dir models/rl")
    
    print("\n=== Example Complete ===")
    print("Generated files:")
    print("  - data/raw/mhqa/sample_*.traj.json (raw trajectories)")
    print("  - data/unified/mhqa/sample_*.json (unified format)")
    print("  - data/rewards/sample_analysis.json (reward analysis)")
    print("  - data/sft/*/train.jsonl (SFT datasets)")
    print("  - models/sft/*/ (trained SFT models)")
    
    print("\nNext steps:")
    print("  1. Generate more diverse trajectories")
    print("  2. Train RL models on high-reward trajectories")
    print("  3. Evaluate models on test datasets")
    print("  4. Use the comprehensive training script: python scripts/train_models.py")


if __name__ == "__main__":
    run_example()
