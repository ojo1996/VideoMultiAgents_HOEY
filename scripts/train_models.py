#!/usr/bin/env python3
"""
Comprehensive Training Script for SFT and RL

This script provides a unified interface for training both SFT and RL models
on agent trajectories with proper data preprocessing and evaluation.
"""

import os
import json
import argparse
import subprocess
import glob
from pathlib import Path
from typing import Dict, Any, List
import yaml


def run_command(cmd: List[str], cwd: str = None) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
    return result


def generate_trajectories(config: Dict[str, Any]) -> List[str]:
    """Generate trajectories using the agent systems."""
    print("=== Generating Trajectories ===")
    
    trajectory_files = []
    agents = config.get('agents', {})
    
    for agent_name, agent_config in agents.items():
        print(f"Generating trajectories for {agent_name}...")
        
        # Create output directory
        output_dir = f"data/raw/{agent_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate multiple trajectories
        num_trajectories = agent_config.get('num_trajectories', 10)
        questions = agent_config.get('questions', [])
        
        for i in range(num_trajectories):
            if i < len(questions):
                question = questions[i]
            else:
                question = f"Sample question {i+1} for {agent_name}"
            
            output_file = f"{output_dir}/traj_{i+1:03d}.traj.json"
            
            # Build command based on agent type
            if agent_name == 'mhqa':
                cmd = [
                    'python', 'agent_systems/MHQA_agent/Main.py',
                    '--question', question,
                    '--out', output_file
                ]
            elif agent_name == 'video':
                video_ctx = agent_config.get('video_context', 'data/video_samples/toy_captions.txt')
                cmd = [
                    'python', 'agent_systems/Video_multiagent/Main.py',
                    '--video_ctx', video_ctx,
                    '--question', question,
                    '--out', output_file
                ]
            elif agent_name == 'math':
                cmd = [
                    'python', 'agent_systems/Math_agent/Main.py',
                    '--question', question,
                    '--out', output_file
                ]
            elif agent_name == 'swe':
                cmd = [
                    'python', 'agent_systems/SWE_agent/Main.py',
                    '--question', question,
                    '--out', output_file
                ]
            elif agent_name == 'tau':
                cmd = [
                    'python', 'agent_systems/TAU_agent/Main.py',
                    '--question', question,
                    '--out', output_file
                ]
            else:
                print(f"Unknown agent: {agent_name}")
                continue
            
            # Run the agent
            result = run_command(cmd)
            if result.returncode == 0:
                trajectory_files.append(output_file)
                print(f"  Generated: {output_file}")
            else:
                print(f"  Failed to generate: {output_file}")
    
    print(f"Generated {len(trajectory_files)} trajectories")
    return trajectory_files


def convert_trajectories(config: Dict[str, Any]) -> List[str]:
    """Convert raw trajectories to unified format."""
    print("=== Converting Trajectories ===")
    
    unified_files = []
    raw_glob = config.get('raw_glob', 'data/raw/*/*.traj.json')
    unified_dir = config.get('unified_dir', 'data/unified')
    
    # Find all trajectory files
    trajectory_files = glob.glob(raw_glob)
    print(f"Found {len(trajectory_files)} trajectory files")
    
    for traj_file in trajectory_files:
        # Determine domain from path
        if '/mhqa/' in traj_file:
            domain = 'mhqa'
        elif '/video/' in traj_file:
            domain = 'video'
        elif '/math/' in traj_file:
            domain = 'math'
        elif '/swe/' in traj_file:
            domain = 'swe'
        elif '/tau/' in traj_file:
            domain = 'tau'
        else:
            domain = 'unknown'
        
        # Create output path
        output_dir = f"{unified_dir}/{domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/{Path(traj_file).stem}.json"
        
        # Convert using the conversion script
        cmd = [
            'python', 'convert_to_out_traj_format.py',
            traj_file,
            '--domain', domain,
            '--out', output_file
        ]
        
        result = run_command(cmd)
        if result.returncode == 0:
            unified_files.append(output_file)
            print(f"  Converted: {traj_file} -> {output_file}")
        else:
            print(f"  Failed to convert: {traj_file}")
    
    print(f"Converted {len(unified_files)} trajectories")
    return unified_files


def build_sft_datasets(config: Dict[str, Any]) -> str:
    """Build SFT datasets from unified trajectories."""
    print("=== Building SFT Datasets ===")
    
    unified_glob = config.get('unified_glob', 'data/unified/*/*.json')
    templates = config.get('templates', 'configs/sft_templates.yaml')
    out_dir = config.get('sft_data_dir', 'data/sft')
    
    cmd = [
        'python', 'scripts/build_sft_datasets.py',
        '--unified_glob', unified_glob,
        '--templates', templates,
        '--out_dir', out_dir
    ]
    
    result = run_command(cmd)
    if result.returncode == 0:
        print(f"SFT datasets built in {out_dir}")
        return out_dir
    else:
        print("Failed to build SFT datasets")
        return None


def train_sft_models(config: Dict[str, Any], sft_data_dir: str) -> List[str]:
    """Train SFT models for each tool."""
    print("=== Training SFT Models ===")
    
    trained_models = []
    tools = config.get('tools', ['reason', 'bash', 'finalize'])
    model_name = config.get('model_name', 'Qwen/Qwen2.5-0.5B')
    out_dir = config.get('sft_models_dir', 'models/sft')
    
    for tool in tools:
        print(f"Training SFT model for {tool}...")
        
        tool_out_dir = f"{out_dir}/{tool}"
        os.makedirs(tool_out_dir, exist_ok=True)
        
        cmd = [
            'python', 'training/trl_train_tool_sft.py',
            '--config', 'configs/trl_defaults.yaml',
            '--model_name', model_name,
            '--tool', tool,
            '--data_root', sft_data_dir,
            '--out_dir', out_dir
        ]
        
        result = run_command(cmd)
        if result.returncode == 0:
            trained_models.append(tool_out_dir)
            print(f"  Trained: {tool_out_dir}")
        else:
            print(f"  Failed to train: {tool}")
    
    print(f"Trained {len(trained_models)} SFT models")
    return trained_models


def calculate_rewards(config: Dict[str, Any]) -> str:
    """Calculate rewards for trajectories."""
    print("=== Calculating Rewards ===")
    
    trajectory_glob = config.get('raw_glob', 'data/raw/*/*.traj.json')
    output_file = config.get('rewards_file', 'data/rewards/analysis.json')
    
    cmd = [
        'python', 'scripts/calculate_rewards.py',
        '--trajectory_glob', trajectory_glob,
        '--output', output_file,
        '--stats'
    ]
    
    result = run_command(cmd)
    if result.returncode == 0:
        print(f"Rewards calculated and saved to {output_file}")
        return output_file
    else:
        print("Failed to calculate rewards")
        return None


def train_rl_models(config: Dict[str, Any]) -> str:
    """Train RL models on trajectories."""
    print("=== Training RL Models ===")
    
    model_name = config.get('model_name', 'Qwen/Qwen2.5-0.5B')
    out_dir = config.get('rl_models_dir', 'models/rl')
    trajectory_glob = config.get('raw_glob', 'data/raw/*/*.traj.json')
    min_reward = config.get('min_reward', 0.0)
    max_examples = config.get('max_examples', 1000)
    
    cmd = [
        'python', 'training/trl_train_rl.py',
        '--config', 'configs/rl_defaults.yaml',
        '--model_name', model_name,
        '--trajectory_glob', trajectory_glob,
        '--out_dir', out_dir,
        '--min_reward', str(min_reward),
        '--max_examples', str(max_examples)
    ]
    
    result = run_command(cmd)
    if result.returncode == 0:
        print(f"RL model trained and saved to {out_dir}")
        return out_dir
    else:
        print("Failed to train RL model")
        return None


def evaluate_models(config: Dict[str, Any], sft_models: List[str], rl_model: str = None) -> Dict[str, Any]:
    """Evaluate trained models."""
    print("=== Evaluating Models ===")
    
    results = {}
    
    # Evaluate SFT models
    for model_path in sft_models:
        tool = Path(model_path).name
        print(f"Evaluating SFT model for {tool}...")
        
        # Run evaluation (you might need to implement this based on your eval.py)
        cmd = ['python', 'eval.py', '--model_path', model_path]
        result = run_command(cmd)
        
        if result.returncode == 0:
            results[f'sft_{tool}'] = {'status': 'success', 'output': result.stdout}
        else:
            results[f'sft_{tool}'] = {'status': 'failed', 'error': result.stderr}
    
    # Evaluate RL model
    if rl_model:
        print(f"Evaluating RL model...")
        cmd = ['python', 'eval.py', '--model_path', rl_model]
        result = run_command(cmd)
        
        if result.returncode == 0:
            results['rl'] = {'status': 'success', 'output': result.stdout}
        else:
            results['rl'] = {'status': 'failed', 'error': result.stderr}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive training script for SFT and RL")
    parser.add_argument("--config", default="configs/training_config.yaml", 
                       help="Training configuration file")
    parser.add_argument("--mode", choices=['sft', 'rl', 'both'], default='both',
                       help="Training mode")
    parser.add_argument("--skip_generation", action='store_true',
                       help="Skip trajectory generation")
    parser.add_argument("--skip_conversion", action='store_true',
                       help="Skip trajectory conversion")
    parser.add_argument("--skip_sft", action='store_true',
                       help="Skip SFT training")
    parser.add_argument("--skip_rl", action='store_true',
                       help="Skip RL training")
    parser.add_argument("--evaluate", action='store_true',
                       help="Evaluate trained models")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Configuration file {args.config} not found, using defaults")
        config = {}
    
    # Override with CLI args
    if args.mode == 'sft':
        args.skip_rl = True
    elif args.mode == 'rl':
        args.skip_sft = True
    
    print(f"Starting training in mode: {args.mode}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Step 1: Generate trajectories
    if not args.skip_generation:
        trajectory_files = generate_trajectories(config)
        if not trajectory_files:
            print("No trajectories generated, exiting")
            return
    else:
        print("Skipping trajectory generation")
        trajectory_files = glob.glob(config.get('raw_glob', 'data/raw/*/*.traj.json'))
    
    # Step 2: Convert trajectories
    if not args.skip_conversion:
        unified_files = convert_trajectories(config)
        if not unified_files:
            print("No trajectories converted, exiting")
            return
    else:
        print("Skipping trajectory conversion")
    
    # Step 3: Calculate rewards
    rewards_file = calculate_rewards(config)
    
    # Step 4: Train SFT models
    sft_models = []
    if not args.skip_sft:
        sft_data_dir = build_sft_datasets(config)
        if sft_data_dir:
            sft_models = train_sft_models(config, sft_data_dir)
    else:
        print("Skipping SFT training")
    
    # Step 5: Train RL models
    rl_model = None
    if not args.skip_rl:
        rl_model = train_rl_models(config)
    else:
        print("Skipping RL training")
    
    # Step 6: Evaluate models
    if args.evaluate:
        results = evaluate_models(config, sft_models, rl_model)
        
        # Save evaluation results
        results_file = 'results/training_results.json'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {results_file}")
    
    print("Training pipeline completed!")


if __name__ == "__main__":
    main()
