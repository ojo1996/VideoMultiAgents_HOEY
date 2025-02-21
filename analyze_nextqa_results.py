import json
from pathlib import Path
import numpy as np
import random

def load_json(filepath):
    """Load JSON file and return the data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_accuracy(data):
    """Calculate accuracy from predictions and truth values."""
    correct = sum(1 for item in data.values() if item['pred'] == item['truth'])
    return correct / len(data) if data else 0

def analyze_nextqa_validation():
    # Define file paths
    base_path = Path('data/nextqa')
    configs = ['multi_star_all', 'single_text', 'single_graph', 'single_video', 'single_all']
    file_paths = {
        config: base_path / f'val_{config}.json'
        for config in configs
    }
    
    # Load all JSON files and sample questions
    data = {}
    sampled_data = {}
    
    for config, filepath in file_paths.items():
        try:
            data[config] = load_json(filepath)
            # Filter questions with pred field
            valid_questions = {k: v for k, v in data[config].items() if 'pred' in v and v['pred'] >= 0}
            
            if len(valid_questions) < 300:
                print(f"Warning: {config} configuration has fewer than 300 valid questions ({len(valid_questions)})")
                sampled_data[config] = valid_questions
            else:
                # Randomly sample 300 questions
                sampled_qids = random.sample(list(valid_questions.keys()), 300)
                sampled_data[config] = {qid: valid_questions[qid] for qid in sampled_qids}
                
        except FileNotFoundError:
            print(f"Warning: Could not find file for {config} configuration")
            continue
    
    # Calculate accuracy for sampled questions for each configuration
    results = {}
    for config in configs:
        if config not in sampled_data:
            continue
            
        accuracy = calculate_accuracy(sampled_data[config])
        results[config] = accuracy
        
        print(f"\n{config.upper()} Configuration:")
        print(f"Number of sampled questions: {len(sampled_data[config])}")
        print(f"Accuracy on sampled questions: {accuracy:.2%}")
        
        # Print breakdown of correct/incorrect counts
        correct = sum(1 for item in sampled_data[config].values() if item['pred'] == item['truth'])
        print(f"Correct: {correct}/{len(sampled_data[config])}")
    
    # Find best and worst performing configs
    if results:
        best_config = max(results.items(), key=lambda x: x[1])
        worst_config = min(results.items(), key=lambda x: x[1])
        
        print(f"\nBest performing configuration: {best_config[0].upper()} ({best_config[1]:.2%})")
        print(f"Worst performing configuration: {worst_config[0].upper()} ({worst_config[1]:.2%})")
        
        # Calculate performance difference between best and worst
        diff = best_config[1] - worst_config[1]
        print(f"Performance gap: {diff:.2%}")
    else:
        print("\nNo results to compare - no valid configurations found")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    analyze_nextqa_validation()
