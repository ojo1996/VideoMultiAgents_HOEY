#!/usr/bin/env python3
"""
RunPod Training Demo

This script demonstrates the RL and SFT training pipeline on RunPod.
It generates sample trajectories and shows how to train models.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print("❌ FAILED")
            print("Error:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def main():
    print("🚀 RunPod Training Demo")
    print("=" * 50)
    
    # Check environment
    print("\n📋 Environment Check:")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files available: {len(list(Path('.').glob('*')))}")
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    os.makedirs("data/raw/mhqa", exist_ok=True)
    os.makedirs("data/unified/mhqa", exist_ok=True)
    os.makedirs("data/sft", exist_ok=True)
    os.makedirs("data/rewards", exist_ok=True)
    os.makedirs("models/sft", exist_ok=True)
    os.makedirs("models/rl", exist_ok=True)
    print("✅ Directories created")
    
    # Step 1: Generate sample trajectories
    print("\n🎯 Step 1: Generating Sample Trajectories")
    questions = [
        "Who wrote The Hobbit and what other famous works did they create?",
        "What is the capital of France and what is its population?",
        "Who was the first person to walk on the moon and when did it happen?"
    ]
    
    trajectory_files = []
    for i, question in enumerate(questions):
        output_file = f"data/raw/mhqa/sample_{i+1}.traj.json"
        
        success = run_command([
            "python", "agent_systems/MHQA_agent/Main.py",
            "--question", question,
            "--out", output_file
        ], f"Generating trajectory {i+1}")
        
        if success and os.path.exists(output_file):
            trajectory_files.append(output_file)
            print(f"✅ Generated: {output_file}")
        else:
            print(f"❌ Failed to generate: {output_file}")
    
    if not trajectory_files:
        print("❌ No trajectories generated. Exiting.")
        return
    
    print(f"✅ Generated {len(trajectory_files)} trajectories")
    
    # Step 2: Convert trajectories to unified format
    print("\n🔄 Step 2: Converting Trajectories")
    unified_files = []
    
    for traj_file in trajectory_files:
        output_file = f"data/unified/mhqa/{Path(traj_file).stem}.json"
        
        success = run_command([
            "python", "convert_to_out_traj_format.py",
            traj_file,
            "--domain", "mhqa",
            "--out", output_file
        ], f"Converting {traj_file}")
        
        if success and os.path.exists(output_file):
            unified_files.append(output_file)
            print(f"✅ Converted: {output_file}")
        else:
            print(f"❌ Failed to convert: {traj_file}")
    
    print(f"✅ Converted {len(unified_files)} trajectories")
    
    # Step 3: Calculate rewards
    print("\n🏆 Step 3: Calculating Rewards")
    success = run_command([
        "python", "scripts/calculate_rewards.py",
        "--trajectory_glob", "data/raw/mhqa/*.traj.json",
        "--output", "data/rewards/analysis.json",
        "--stats"
    ], "Calculating rewards")
    
    if success:
        print("✅ Rewards calculated")
        # Show reward stats
        try:
            with open("data/rewards/analysis.json", 'r') as f:
                data = json.load(f)
            if data:
                rewards = [item['rewards']['total'] for item in data]
                print(f"   Average reward: {sum(rewards)/len(rewards):.3f}")
                print(f"   Min reward: {min(rewards):.3f}")
                print(f"   Max reward: {max(rewards):.3f}")
        except:
            pass
    else:
        print("❌ Failed to calculate rewards")
    
    # Step 4: Build SFT datasets
    print("\n📚 Step 4: Building SFT Datasets")
    success = run_command([
        "python", "scripts/build_sft_datasets.py",
        "--unified_glob", "data/unified/*/*.json",
        "--templates", "configs/sft_templates.yaml",
        "--out_dir", "data/sft"
    ], "Building SFT datasets")
    
    if success:
        print("✅ SFT datasets built")
        # Show dataset stats
        try:
            with open("data/sft/SUMMARY.json", 'r') as f:
                summary = json.load(f)
            print("   Dataset summary:")
            for tool, stats in summary.items():
                print(f"     {tool}: {stats['train']} train, {stats['val']} val")
        except:
            pass
    else:
        print("❌ Failed to build SFT datasets")
    
    # Step 5: Train a simple SFT model (if we have data)
    print("\n🎓 Step 5: Training SFT Model")
    
    # Check if we have SFT data
    sft_files = list(Path("data/sft").glob("*/train.jsonl"))
    if not sft_files:
        print("⚠️  No SFT data found, skipping training")
    else:
        tool = sft_files[0].parent.name
        print(f"   Training on tool: {tool}")
        
        # Use a smaller model for CPU training
        success = run_command([
            "python", "training/trl_train_tool_sft.py",
            "--config", "configs/trl_defaults.yaml",
            "--model_name", "Qwen/Qwen2.5-0.5B",  # Smaller model for CPU
            "--tool", tool,
            "--data_root", "data/sft",
            "--out_dir", "models/sft"
        ], f"Training SFT model for {tool}")
        
        if success:
            print(f"✅ SFT model trained for {tool}")
        else:
            print(f"❌ Failed to train SFT model for {tool}")
    
    # Step 6: Show RL training command (without actually running it)
    print("\n🤖 Step 6: RL Training (Example Command)")
    print("To train RL models, you would run:")
    print("python training/trl_train_rl.py \\")
    print("    --config configs/rl_defaults.yaml \\")
    print("    --model_name Qwen/Qwen2.5-0.5B \\")
    print("    --trajectory_glob 'data/raw/*/*.traj.json' \\")
    print("    --out_dir models/rl \\")
    print("    --min_reward 0.3")
    
    # Summary
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("Generated files:")
    print(f"  - Raw trajectories: {len(trajectory_files)} files")
    print(f"  - Unified format: {len(unified_files)} files")
    print(f"  - SFT datasets: {len(sft_files)} tools")
    print("  - Reward analysis: data/rewards/analysis.json")
    print("  - SFT models: models/sft/")
    
    print("\nNext steps:")
    print("1. Generate more diverse trajectories")
    print("2. Train RL models on high-reward trajectories")
    print("3. Evaluate models on test datasets")
    print("4. Use the comprehensive training script:")
    print("   python scripts/train_models.py --mode both")
    
    print("\nFor GPU training on RunPod:")
    print("1. Ensure CUDA is available")
    print("2. Use larger models (Qwen2.5-7B-Instruct)")
    print("3. Increase batch sizes in configs")
    print("4. Use the demo_runpod.sh script for evaluation")

if __name__ == "__main__":
    main()
