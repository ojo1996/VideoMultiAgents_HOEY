#!/usr/bin/env python3
"""
Master SWE SFT Training Pipeline
Orchestrates the entire process: generate trajectories -> build datasets -> train models
"""

import os
import json
import argparse
import subprocess
import glob
from pathlib import Path
from typing import List, Dict, Any

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_file_exists(file_path: str, description: str) -> bool:
    """Check if file exists and print status."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (not found)")
        return False

def main():
    parser = argparse.ArgumentParser(description="Master SWE SFT Training Pipeline")
    parser.add_argument("--num_trajectories", type=int, default=10, help="Number of trajectories to generate")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B", help="Base model for training")
    parser.add_argument("--output_base", default="data", help="Base output directory")
    parser.add_argument("--skip_generation", action="store_true", help="Skip trajectory generation")
    parser.add_argument("--skip_dataset_building", action="store_true", help="Skip dataset building")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    args = parser.parse_args()
    
    # Set up paths
    raw_dir = os.path.join(args.output_base, "raw", "swe")
    sft_dir = os.path.join(args.output_base, "sft", "swe")
    models_dir = os.path.join(args.output_base, "models", "swe")
    
    print("🚀 Starting SWE SFT Training Pipeline")
    print(f"📁 Raw trajectories: {raw_dir}")
    print(f"📁 SFT datasets: {sft_dir}")
    print(f"📁 Trained models: {models_dir}")
    
    # Step 1: Generate trajectories
    if not args.skip_generation:
        print("\n" + "="*50)
        print("STEP 1: Generate SWE Agent Trajectories")
        print("="*50)
        
        success = run_command([
            "python", "scripts/generate_swe_trajectories.py",
            "--output_dir", raw_dir,
            "--num_samples", str(args.num_trajectories)
        ], "Generate SWE trajectories")
        
        if not success:
            print("❌ Trajectory generation failed. Exiting.")
            return
        
        # Check if trajectories were generated
        traj_files = glob.glob(os.path.join(raw_dir, "*.traj.json"))
        if len(traj_files) == 0:
            print("❌ No trajectory files generated. Exiting.")
            return
        
        print(f"✅ Generated {len(traj_files)} trajectory files")
    
    # Step 2: Build SFT datasets
    if not args.skip_dataset_building:
        print("\n" + "="*50)
        print("STEP 2: Build Tool-Specific SFT Datasets")
        print("="*50)
        
        success = run_command([
            "python", "scripts/build_swe_sft_datasets.py",
            "--trajectory_glob", os.path.join(raw_dir, "*.traj.json"),
            "--output_dir", sft_dir
        ], "Build SFT datasets")
        
        if not success:
            print("❌ Dataset building failed. Exiting.")
            return
        
        # Check if datasets were created
        bash_data = os.path.join(sft_dir, "swe_bash_sft.jsonl")
        file_edit_data = os.path.join(sft_dir, "swe_file_edit_sft.jsonl")
        
        if not (check_file_exists(bash_data, "Bash dataset") and 
                check_file_exists(file_edit_data, "File edit dataset")):
            print("❌ Required datasets not found. Exiting.")
            return
    
    # Step 3: Train models
    if not args.skip_training:
        print("\n" + "="*50)
        print("STEP 3: Train Tool-Specific Models")
        print("="*50)
        
        # Train bash model
        bash_model_dir = os.path.join(models_dir, "bash")
        bash_data = os.path.join(sft_dir, "swe_bash_sft.jsonl")
        
        if os.path.exists(bash_data):
            print(f"\n🔧 Training bash model...")
            success = run_command([
                "python", "training/sft_train_swe_tool.py",
                "--model_name", args.model_name,
                "--tool", "bash",
                "--data_file", bash_data,
                "--out_dir", bash_model_dir,
                "--num_epochs", "2",
                "--batch_size", "1"
            ], "Train bash model")
            
            if success:
                print(f"✅ Bash model saved to: {bash_model_dir}")
            else:
                print("❌ Bash model training failed")
        else:
            print("❌ Bash dataset not found, skipping bash model training")
        
        # Train file_edit model
        file_edit_model_dir = os.path.join(models_dir, "file_edit")
        file_edit_data = os.path.join(sft_dir, "swe_file_edit_sft.jsonl")
        
        if os.path.exists(file_edit_data):
            print(f"\n📝 Training file_edit model...")
            success = run_command([
                "python", "training/sft_train_swe_tool.py",
                "--model_name", args.model_name,
                "--tool", "file_edit",
                "--data_file", file_edit_data,
                "--out_dir", file_edit_model_dir,
                "--num_epochs", "2",
                "--batch_size", "1"
            ], "Train file_edit model")
            
            if success:
                print(f"✅ File edit model saved to: {file_edit_model_dir}")
            else:
                print("❌ File edit model training failed")
        else:
            print("❌ File edit dataset not found, skipping file_edit model training")
    
    # Final summary
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    
    # Count trajectories
    traj_files = glob.glob(os.path.join(raw_dir, "*.traj.json"))
    print(f"📊 Trajectories generated: {len(traj_files)}")
    
    # Count datasets
    dataset_files = glob.glob(os.path.join(sft_dir, "*.jsonl"))
    print(f"📊 SFT datasets created: {len(dataset_files)}")
    
    # Count models
    model_dirs = [d for d in glob.glob(os.path.join(models_dir, "*")) if os.path.isdir(d)]
    print(f"📊 Models trained: {len(model_dirs)}")
    
    # List all outputs
    print(f"\n📁 Outputs:")
    print(f"  Raw trajectories: {raw_dir}")
    print(f"  SFT datasets: {sft_dir}")
    print(f"  Trained models: {models_dir}")
    
    print("\n🎉 SWE SFT Pipeline completed!")

if __name__ == "__main__":
    main()
