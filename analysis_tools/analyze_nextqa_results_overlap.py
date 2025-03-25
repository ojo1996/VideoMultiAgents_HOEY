import json
from pathlib import Path
import numpy as np
import random
from functools import reduce
from typing import Dict, Any

def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file and return the data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_accuracy(data: Dict[str, Dict[str, Any]]) -> float:
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
    
    # Load all JSON files
    data = {}
    valid_questions = {}
    
    # First, get valid questions for each configuration
    for config, filepath in file_paths.items():
        try:
            data[config] = load_json(filepath)
            # Filter questions with pred field
            valid_questions[config] = {
                k: v for k, v in data[config].items() 
                if 'pred' in v and v['pred'] >= 0
            }
            print(f"{config}: {len(valid_questions[config])} valid questions")
        except FileNotFoundError:
            print(f"Warning: Could not find file for {config} configuration")
            valid_questions[config] = {}
    
    # Find question IDs that appear in all configurations
    if not valid_questions:
        print("No valid configurations found")
        return
        
    # Get sets of question IDs for each configuration
    question_sets = [set(conf_data.keys()) for conf_data in valid_questions.values()]
    
    # Find intersection of all sets
    common_questions = reduce(lambda x, y: x & y, question_sets)
    
    print(f"\nNumber of questions common to all configurations: {len(common_questions)}")
    
    # Calculate accuracy for common questions for each configuration
    results = {}
    for config in configs:
        if config not in valid_questions:
            continue
            
        # Filter to only common questions
        common_data = {
            qid: valid_questions[config][qid] 
            for qid in common_questions
        }
        
        accuracy = calculate_accuracy(common_data)
        results[config] = accuracy
        
        print(f"\n{config.upper()} Configuration:")
        print(f"Accuracy on common questions: {accuracy:.2%}")
        
        # Print breakdown of correct/incorrect counts
        correct = sum(1 for item in common_data.values() if item['pred'] == item['truth'])
        print(f"Correct: {correct}/{len(common_data)}")
    
    # Find best and worst performing configs
    if results:
        best_config = max(results.items(), key=lambda x: x[1])
        worst_config = min(results.items(), key=lambda x: x[1])
        
        print(f"\nBest performing configuration: {best_config[0].upper()} ({best_config[1]:.2%})")
        print(f"Worst performing configuration: {worst_config[0].upper()} ({worst_config[1]:.2%})")
        
        # Calculate performance difference between best and worst
        diff = best_config[1] - worst_config[1]
        print(f"Performance gap: {diff:.2%}")
        
        # Calculate average performance across all configurations
        avg_performance = sum(results.values()) / len(results)
        print(f"Average performance across configurations: {avg_performance:.2%}")

if __name__ == "__main__":
    analyze_nextqa_validation()
