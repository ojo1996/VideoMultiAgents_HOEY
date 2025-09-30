#!/usr/bin/env python3
"""
Fixed RL Training Script for Trajectories using PPO
"""

import os
import json
import argparse
import random
import glob
from typing import Dict, Any, List, Optional
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import yaml

def load_trajectory(path: str) -> Dict[str, Any]:
    """Load a trajectory JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_reward(trajectory: Dict[str, Any], domain: str) -> float:
    """Calculate reward for a trajectory."""
    rewards = {}
    
    # 1. Task completion reward
    success = trajectory.get('success', False)
    final_answer = trajectory.get('final_answer', '')
    rewards['completion'] = 1.0 if success else (0.5 if final_answer and len(final_answer.strip()) > 5 else 0.0)
    
    # 2. Tool usage efficiency
    actions = trajectory.get('actions', [])
    tool_actions = [a for a in actions if a.get('tool') not in ['reason', 'finalize', 'decompose', 'reason_hop']]
    tool_count = len(tool_actions)
    rewards['efficiency'] = max(0.0, 1.0 - (tool_count - 1) * 0.1) if tool_count > 0 else 0.5
    
    # 3. Answer quality
    rewards['quality'] = 0.7 if final_answer and len(final_answer.strip()) > 10 else 0.3
    
    # 4. Reasoning quality
    reasoning_actions = [a for a in actions if a.get('tool') in ['reason', 'reason_hop']]
    rewards['reasoning'] = 0.8 if len(reasoning_actions) > 0 else 0.2
    
    # Weighted total
    weights = {'completion': 0.4, 'efficiency': 0.2, 'quality': 0.3, 'reasoning': 0.1}
    rewards['total'] = sum(rewards[key] * weights[key] for key in weights)
    
    return rewards['total']

def trajectory_to_conversation(trajectory: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert trajectory to conversation format for RL training."""
    messages = []
    question = trajectory.get('question', '')
    
    if not question:
        return messages
    
    # Add initial user message
    messages.append({
        "role": "user",
        "content": f"Task: {question}\n\nPlease solve this step by step."
    })
    
    # Convert actions to conversation
    actions = trajectory.get('actions', [])
    for action in actions:
        tool = action.get('tool', '')
        output = action.get('output', '')
        
        if tool == 'reason':
            messages.append({
                "role": "assistant",
                "content": f"Let me think about this:\n{output}"
            })
        elif tool in ['bash', 'load_context', 'reason_hop']:
            input_data = action.get('input', {})
            if isinstance(input_data, dict):
                input_str = str(input_data)
            else:
                input_str = str(input_data)
            
            messages.append({
                "role": "assistant", 
                "content": f"I'll use {tool}:\nInput: {input_str}\nOutput: {output}"
            })
        elif tool == 'finalize':
            messages.append({
                "role": "assistant",
                "content": f"Final answer: {output}"
            })
    
    return messages

def create_rl_dataset(trajectory_paths: List[str], min_reward: float = 0.0) -> Dataset:
    """Create RL dataset from trajectory files."""
    dataset_data = []
    
    for traj_path in trajectory_paths:
        try:
            trajectory = load_trajectory(traj_path)
            domain = trajectory.get('domain', 'unknown')
            
            # Calculate reward
            reward = calculate_reward(trajectory, domain)
            
            # Skip low-reward trajectories if filtering
            if reward < min_reward:
                continue
            
            # Convert to conversation format
            messages = trajectory_to_conversation(trajectory)
            
            if not messages:
                continue
            
            dataset_data.append({
                "messages": messages,
                "reward": reward,
                "task_id": trajectory.get('task_id', 'unknown'),
                "domain": domain,
                "success": trajectory.get('success', False)
            })
            
        except Exception as e:
            print(f"Error processing {traj_path}: {e}")
            continue
    
    print(f"Created RL dataset with {len(dataset_data)} examples")
    print(f"Average reward: {np.mean([d['reward'] for d in dataset_data]):.3f}")
    
    return Dataset.from_list(dataset_data)

def main():
    parser = argparse.ArgumentParser(description="RL Training on Trajectories")
    parser.add_argument("--config", default="configs/rl_defaults.yaml")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--trajectory_glob", default="data/raw/*.traj.json")
    parser.add_argument("--out_dir", default="models/rl")
    parser.add_argument("--min_reward", type=float, default=0.0)
    parser.add_argument("--max_examples", type=int, default=1000)
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Set random seeds
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Find trajectory files
    trajectory_paths = glob.glob(args.trajectory_glob)
    if not trajectory_paths:
        print(f"No trajectory files found matching {args.trajectory_glob}")
        return
    
    print(f"Found {len(trajectory_paths)} trajectory files")
    
    # Create dataset
    dataset = create_rl_dataset(trajectory_paths, args.min_reward)
    
    if len(dataset) == 0:
        print("No valid trajectories found for training")
        return
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
    
    # PPO Configuration (FIXED - removed model_name)
    ppo_config = PPOConfig(
        learning_rate=config.get('learning_rate', 1e-5),
        batch_size=config.get('batch_size', 4),
        mini_batch_size=config.get('mini_batch_size', 2),
        ppo_epochs=config.get('ppo_epochs', 4),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        target_kl=config.get('target_kl', 0.1),
        vf_coef=config.get('vf_coef', 0.1),
        cliprange=config.get('cliprange', 0.2),
        cliprange_value=config.get('cliprange_value', 0.2),
        gamma=config.get('gamma', 0.99),
        lam=config.get('lam', 0.95),
        log_with=config.get('log_with', 'tensorboard'),
        project_kwargs={"logging_dir": os.path.join(args.out_dir, "logs")},
    )
    
    # Create trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=config.get('num_train_epochs', 3),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 1),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        learning_rate=config.get('learning_rate', 1e-5),
        warmup_ratio=config.get('warmup_ratio', 0.1),
        logging_steps=config.get('logging_steps', 10),
        save_steps=config.get('save_steps', 100),
        eval_steps=config.get('eval_steps', 100),
        save_total_limit=config.get('save_total_limit', 3),
        load_best_model_at_end=True,
        metric_for_best_model="reward",
        greater_is_better=True,
        report_to=config.get('report_to', ['tensorboard']),
        remove_unused_columns=False,
    )
    
    # Train
    print("Starting RL training...")
    trainer.train()
    
    # Save model
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    
    # Save training metrics
    metrics = trainer.evaluate()
    with open(os.path.join(args.out_dir, "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"RL training completed. Model saved to {args.out_dir}")
    print(f"Final metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    main()
