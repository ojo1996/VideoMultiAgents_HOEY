#!/usr/bin/env python3
"""
RL Training Script for Trajectories using PPO

This script implements reinforcement learning training on agent trajectories
using PPO (Proximal Policy Optimization) from the TRL library.
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
    """
    Calculate reward for a trajectory based on domain-specific criteria.
    
    Args:
        trajectory: The trajectory dictionary
        domain: The domain (mhqa, video, math, swe, tau)
    
    Returns:
        Total reward score
    """
    rewards = {}
    
    # 1. Task completion reward
    success = trajectory.get('success', False)
    rewards['completion'] = 1.0 if success else 0.0
    
    # 2. Tool usage efficiency (penalize excessive tool usage)
    actions = trajectory.get('actions', [])
    tool_actions = [a for a in actions if a.get('tool') not in ['reason', 'finalize']]
    tool_count = len(tool_actions)
    rewards['efficiency'] = max(0.0, 1.0 - (tool_count - 1) * 0.1)  # Optimal: 1-2 tools
    
    # 3. Answer quality (domain-specific)
    final_answer = trajectory.get('final_answer', '')
    rewards['quality'] = evaluate_answer_quality(final_answer, domain)
    
    # 4. Reasoning quality
    reasoning_actions = [a for a in actions if a.get('tool') == 'reason']
    rewards['reasoning'] = evaluate_reasoning_quality(reasoning_actions)
    
    # 5. Trajectory length penalty (encourage concise solutions)
    total_length = sum(len(str(a.get('output', ''))) for a in actions)
    rewards['conciseness'] = max(0.0, 1.0 - total_length / 10000)  # Penalty for very long outputs
    
    # Weighted sum
    weights = {
        'completion': 0.4,
        'efficiency': 0.2,
        'quality': 0.3,
        'reasoning': 0.1,
        'conciseness': 0.0  # Optional penalty
    }
    
    total_reward = sum(rewards[key] * weights[key] for key in weights)
    return total_reward


def evaluate_answer_quality(answer: str, domain: str) -> float:
    """Evaluate answer quality based on domain-specific criteria."""
    if not answer or len(answer.strip()) < 5:
        return 0.0
    
    score = 0.5  # Base score
    
    # Domain-specific quality checks
    if domain == 'math':
        # Check for mathematical reasoning indicators
        math_indicators = ['equation', 'solve', 'calculate', 'formula', 'theorem']
        if any(indicator in answer.lower() for indicator in math_indicators):
            score += 0.3
        if any(char.isdigit() for char in answer):
            score += 0.2
    
    elif domain == 'mhqa':
        # Check for multi-hop reasoning
        if 'first' in answer.lower() and 'then' in answer.lower():
            score += 0.3
        if answer.count('.') > 2:  # Multiple sentences suggest detailed reasoning
            score += 0.2
    
    elif domain == 'video':
        # Check for video-specific content
        video_indicators = ['video', 'scene', 'frame', 'action', 'character']
        if any(indicator in answer.lower() for indicator in video_indicators):
            score += 0.3
    
    elif domain == 'swe':
        # Check for code quality
        if '```' in answer or 'def ' in answer or 'class ' in answer:
            score += 0.3
        if 'error' not in answer.lower() and 'bug' not in answer.lower():
            score += 0.2
    
    return min(1.0, score)


def evaluate_reasoning_quality(reasoning_actions: List[Dict[str, Any]]) -> float:
    """Evaluate the quality of reasoning steps."""
    if not reasoning_actions:
        return 0.0
    
    total_score = 0.0
    for action in reasoning_actions:
        output = action.get('output', '')
        if len(output) < 10:  # Too short
            continue
        
        # Check for reasoning indicators
        reasoning_indicators = ['because', 'therefore', 'since', 'thus', 'hence', 'so']
        if any(indicator in output.lower() for indicator in reasoning_indicators):
            total_score += 0.3
        
        # Check for step-by-step structure
        if any(char.isdigit() for char in output) and ('.' in output or '\n' in output):
            total_score += 0.2
        
        # Check for logical connectors
        connectors = ['and', 'or', 'but', 'however', 'moreover', 'furthermore']
        if any(connector in output.lower() for connector in connectors):
            total_score += 0.1
    
    return min(1.0, total_score / len(reasoning_actions))


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
            # Tool usage
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


def collate_fn(batch):
    """Custom collate function for RL training."""
    # This is a simplified version - you might need to implement
    # proper tokenization and padding based on your specific needs
    return {
        "input_ids": torch.tensor([1, 2, 3]),  # Placeholder
        "attention_mask": torch.tensor([1, 1, 1]),  # Placeholder
        "rewards": torch.tensor([item["reward"] for item in batch])
    }


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
    
    # Override with CLI args
    model_name = args.model_name or config.get('model_name', 'Qwen/Qwen2.5-0.5B')
    out_dir = args.out_dir or config.get('out_dir', 'models/rl')
    min_reward = args.min_reward or config.get('min_reward', 0.0)
    max_examples = args.max_examples or config.get('max_examples', 1000)
    
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
    
    # Limit examples if specified
    if len(trajectory_paths) > max_examples:
        trajectory_paths = random.sample(trajectory_paths, max_examples)
        print(f"Limited to {max_examples} examples")
    
    # Create dataset
    dataset = create_rl_dataset(trajectory_paths, min_reward)
    
    if len(dataset) == 0:
        print("No valid trajectories found for training")
        return
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # PPO Configuration
    ppo_config = PPOConfig(
        model_name=model_name,
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
        project_kwargs={"logging_dir": os.path.join(out_dir, "logs")},
    )
    
    # Create trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collate_fn,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
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
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    # Save training metrics
    metrics = trainer.evaluate()
    with open(os.path.join(out_dir, "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"RL training completed. Model saved to {out_dir}")
    print(f"Final metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
