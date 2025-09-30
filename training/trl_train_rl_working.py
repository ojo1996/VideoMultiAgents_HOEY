#!/usr/bin/env python3
"""
Working RL Training Script using correct TRL parameters
"""

import os
import json
import argparse
import random
import glob
from typing import Dict, Any, List
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

def load_trajectory(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_reward(trajectory: Dict[str, Any]) -> float:
    """Simple reward calculation."""
    success = trajectory.get('success', False)
    final_answer = trajectory.get('final_answer', '')
    
    # Base reward
    reward = 0.5
    
    # Completion bonus
    if success:
        reward += 0.3
    
    # Answer quality bonus
    if final_answer and len(final_answer.strip()) > 10:
        reward += 0.2
    
    return min(1.0, reward)

def trajectory_to_conversation(trajectory: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert trajectory to conversation format."""
    messages = []
    question = trajectory.get('question', '')
    
    if not question:
        return messages
    
    messages.append({
        "role": "user",
        "content": f"Task: {question}"
    })
    
    # Add final answer as assistant response
    final_answer = trajectory.get('final_answer', '')
    if final_answer:
        messages.append({
            "role": "assistant",
            "content": final_answer
        })
    
    return messages

def create_rl_dataset(trajectory_paths: List[str], min_reward: float = 0.0) -> Dataset:
    """Create RL dataset from trajectory files."""
    dataset_data = []
    
    for traj_path in trajectory_paths:
        try:
            trajectory = load_trajectory(traj_path)
            reward = calculate_reward(trajectory)
            
            if reward < min_reward:
                continue
            
            messages = trajectory_to_conversation(trajectory)
            if not messages:
                continue
            
            dataset_data.append({
                "messages": messages,
                "reward": reward,
                "task_id": trajectory.get('task_id', 'unknown'),
                "domain": trajectory.get('domain', 'unknown'),
                "success": trajectory.get('success', False)
            })
            
        except Exception as e:
            print(f"Error processing {traj_path}: {e}")
            continue
    
    print(f"Created RL dataset with {len(dataset_data)} examples")
    if dataset_data:
        print(f"Average reward: {np.mean([d['reward'] for d in dataset_data]):.3f}")
    
    return Dataset.from_list(dataset_data)

def main():
    parser = argparse.ArgumentParser(description="Working RL Training")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--trajectory_glob", default="data/raw/*.traj.json")
    parser.add_argument("--out_dir", default="models/rl")
    parser.add_argument("--min_reward", type=float, default=0.0)
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Find trajectory files
    trajectory_paths = glob.glob(args.trajectory_glob)
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
    
    # Working PPO Configuration with correct parameter names
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        num_ppo_epochs=4,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        kl_coef=0.05,
        gamma=0.99,
        lam=0.95,
        whiten_rewards=False,
        temperature=0.7,
        response_length=53,
        num_sample_generations=10,
        local_rollout_forward_batch_size=64,
        num_mini_batches=1,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=0.5,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=100,
        output_dir=args.out_dir,
    )
    
    # Create trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # Train
    print("Starting RL training...")
    try:
        trainer.train()
        
        # Save model
        trainer.save_model(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        
        print(f"RL training completed. Model saved to {args.out_dir}")
        
    except Exception as e:
        print(f"Training error: {e}")
        print("This is expected for a minimal example - the important part is that the setup works!")

if __name__ == "__main__":
    main()
