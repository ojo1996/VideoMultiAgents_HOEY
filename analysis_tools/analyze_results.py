import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_json_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return {}

def check_correct(data):
    results = {}
    for q_id, q_data in data.items():
        # Check if prediction matches truth
        truth = q_data.get('truth')
        pred = q_data.get('pred')
        results[q_id] = 1 if truth == pred else 0
    return results

def create_comparison_matrix():
    # Load all files
    files = {
        'Multi-Vision-Text': 'data/nextqa/test_small_vision_text_simple_multi_agent.json',
        'Single-Vision-Text': 'data/nextqa/test_small_vision_text_simple_single_agent.json',
        'Single-Vision': 'data/nextqa/test_small_vision_simple_single_agent.json',
        'Single-Text': 'data/nextqa/test_small_simple_single_agent.json'
    }
    
    # Create results dictionary
    all_results = {}
    all_questions = set()
    
    # Process each file
    for model_type, filepath in files.items():
        data = load_json_file(filepath)
        results = check_correct(data)
        all_results[model_type] = results
        all_questions.update(results.keys())
    
    # Create DataFrame
    df = pd.DataFrame(index=sorted(all_questions), columns=files.keys())
    
    # Fill DataFrame
    for model_type in files.keys():
        for q_id in all_questions:
            df.at[q_id, model_type] = all_results[model_type].get(q_id, None)
    
    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    print("------------------")
    success_rates = df.mean() * 100
    for model_type, rate in success_rates.items():
        print(f"{model_type}: {rate:.1f}% correct")
    
    return df

if __name__ == "__main__":
    df = create_comparison_matrix()
    print("\nDetailed Results:")
    print(df)
