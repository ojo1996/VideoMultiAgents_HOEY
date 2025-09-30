#!/usr/bin/env python3
"""
SFT Training Script for SWE Tool-Specific Models
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

def load_sft_dataset(jsonl_path: str) -> List[Dict[str, str]]:
    """Load SFT dataset from JSONL file."""
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize the dataset."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    parser = argparse.ArgumentParser(description="SFT Training for SWE Tool-Specific Models")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tool", required=True, choices=["bash", "file_edit"], help="Tool to train")
    parser.add_argument("--data_file", required=True, help="Path to JSONL training data")
    parser.add_argument("--out_dir", required=True, help="Output directory for trained model")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Load dataset
    print(f"Loading {args.tool} dataset from {args.data_file}")
    examples = load_sft_dataset(args.data_file)
    print(f"Loaded {len(examples)} examples")
    
    if len(examples) == 0:
        print("No examples found!")
        return
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=50,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        warmup_steps=5,
        save_total_limit=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_steps=-1,
        save_only_model=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train
    print(f"Starting SFT training for {args.tool} tool...")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Sequence length: {args.max_length}")
    
    try:
        # Clear cache before training
        torch.cuda.empty_cache()
        
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(args.out_dir)
        
        print(f"SFT training completed. Model saved to {args.out_dir}")
        
        # Test the model
        print(f"\nTesting the trained {args.tool} model...")
        if args.tool == "bash":
            test_prompt = "Execute the following bash command for write phase: ls -la"
        else:
            test_prompt = "Perform file editing operation for write phase"
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test prompt: {test_prompt}")
        print(f"Model response: {response}")
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {args.out_dir}")
        print(f"Training examples: {len(examples)}")
        print(f"Epochs: {args.num_epochs}")
        print(f"Learning rate: {args.learning_rate}")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
