#!/usr/bin/env python3
"""
Simple SFT Training Script - Fixed parameter names
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
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

def load_trajectory(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def trajectory_to_sft_format(trajectory: Dict[str, Any]) -> Dict[str, str]:
    """Convert trajectory to SFT format."""
    question = trajectory.get('question', '')
    final_answer = trajectory.get('final_answer', '')
    
    if not question or not final_answer:
        return None
    
    # Create instruction-response pair
    instruction = f"Answer the following question: {question}"
    response = final_answer
    
    # Format as conversation
    text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    
    return {
        "text": text,
        "instruction": instruction,
        "response": response,
        "domain": trajectory.get('domain', 'unknown'),
        "success": trajectory.get('success', False)
    }

def create_sft_dataset(trajectory_paths: List[str]) -> Dataset:
    """Create SFT dataset from trajectory files."""
    dataset_data = []
    
    for traj_path in trajectory_paths:
        try:
            trajectory = load_trajectory(traj_path)
            sft_data = trajectory_to_sft_format(trajectory)
            
            if sft_data is None:
                continue
                
            dataset_data.append(sft_data)
            
        except Exception as e:
            print(f"Error processing {traj_path}: {e}")
            continue
    
    print(f"Created SFT dataset with {len(dataset_data)} examples")
    return Dataset.from_list(dataset_data)

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the dataset."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    parser = argparse.ArgumentParser(description="Simple SFT Training")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--trajectory_glob", default="data/raw/*.traj.json")
    parser.add_argument("--out_dir", default="models/sft")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Find trajectory files
    trajectory_paths = glob.glob(args.trajectory_glob)
    print(f"Found {len(trajectory_paths)} trajectory files")
    
    # Create dataset
    dataset = create_sft_dataset(trajectory_paths)
    
    if len(dataset) == 0:
        print("No valid trajectories found for training")
        return
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments with correct parameter names
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        eval_strategy="no",  # Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting SFT training...")
    try:
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(args.out_dir)
        
        print(f"SFT training completed. Model saved to {args.out_dir}")
        
        # Test the model
        print("\nTesting the trained model...")
        test_prompt = "Answer the following question: What is the capital of France?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test prompt: {test_prompt}")
        print(f"Model response: {response}")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
