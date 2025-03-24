import json
from pathlib import Path
from typing import Dict, Any, Tuple

def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file and return the data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def find_different_predictions(
    config_a: str,
    config_b: str,
    base_path: Path = Path('data/nextqa')
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Find questions where config_a got correct but config_b got wrong.
    
    Args:
        config_a: Name of first configuration (e.g., 'multi_star_all')
        config_b: Name of second configuration (e.g., 'single_text')
        base_path: Base path to the data directory
    
    Returns:
        Tuple containing:
        - Dictionary of cases where config_a succeeded but config_b failed
        - Dictionary of cases where config_b succeeded but config_a failed
    """
    def load_config_data(config_name):
        """Helper function to load config data"""
        config_path = base_path / f'val_{config_name}.json'
        return load_json(config_path)
    # Load JSON files once at function start
    captions_data = None
    graph_data = None
    
    def add_extra_data(result, config_name, q_data):
        """Helper function to add additional data based on config type"""
        nonlocal captions_data, graph_data
        
        if config_name == 'single_text':
            if captions_data is None:
                captions_path = base_path / 'captions_gpt4o.json'
                captions_data = load_json(captions_path)
            vid_id = q_data['map_vid_vidorid'].split('/')[-1]
            if vid_id in captions_data:
                result[config_name]['captions'] = captions_data[vid_id]
                
        elif config_name == 'single_graph':
            if graph_data is None:
                graph_path = base_path / 'nextqa_graph_captions.json'
                graph_data = load_json(graph_path)
            graph_key = q_data['map_vid_vidorid'].replace('/', '-')
            if graph_key in graph_data:
                result[config_name]['scene_graph'] = graph_data[graph_key]

    def create_result_entry(q_data, other_q_data, config_name, other_config_name):
        """Helper function to create result dictionary entry"""
        result = {
            'question': q_data['question'],
            'truth': q_data['truth'],
            'map_vid_vidorid': q_data.get('map_vid_vidorid', ''),
            'type': q_data['type'],
            config_name: {
                'pred': q_data['pred'],
                'prompt': q_data.get('agent_prompts', ''),
                'response': q_data.get('response', '')
            },
            other_config_name: {
                'pred': other_q_data['pred'],
                'prompt': other_q_data.get('agent_prompts', ''),
                'response': other_q_data.get('response', '')
            }
        }

        # Add additional data for both configurations
        add_extra_data(result, config_name, q_data)
        add_extra_data(result, other_config_name, other_q_data)

        return result

    # Load the JSON files
    config_a_data = load_config_data(config_a)
    config_b_data = load_config_data(config_b)
    
    # Dictionaries to store the differences
    a_better_than_b = {}
    b_better_than_a = {}
    
    # Analyze each question
    for qid, config_a_q in config_a_data.items():
        # Skip if question doesn't exist in both datasets or missing pred field
        if (qid not in config_b_data or 
            'pred' not in config_a_q or 
            'pred' not in config_b_data[qid]):
            continue
            
        config_b_q = config_b_data[qid]
        
        # Check if config_a was correct and config_b was wrong
        if (config_a_q['pred'] == config_a_q['truth'] and 
            config_b_q['pred'] != config_b_q['truth']):
            a_better_than_b[qid] = create_result_entry(
                config_a_q, config_b_q, config_a, config_b)
        
        # Check if config_b was correct and config_a was wrong
        elif (config_b_q['pred'] == config_b_q['truth'] and 
              config_a_q['pred'] != config_a_q['truth']):
            b_better_than_a[qid] = create_result_entry(
                config_b_q, config_a_q, config_b, config_a)
    
    return a_better_than_b, b_better_than_a

from itertools import combinations

def main():
    try:
        # Define all configurations to compare
        configs = ['single_text', 'single_video', 'single_graph']
        
        # Compare all unique pairs
        for config_a, config_b in combinations(configs, 2):
            print(f"\nComparing {config_a} vs {config_b}")
            
            # Find the differences
            a_better, b_better = find_different_predictions(config_a, config_b)
            
            # Print summary
            print(f"Found {len(a_better)} questions where {config_a} succeeded but {config_b} failed")
            print(f"Found {len(b_better)} questions where {config_b} succeeded but {config_a} failed")
            
            # Save to JSON files
            output_a_path = f'{config_a}_better_than_{config_b}.json'
            output_b_path = f'{config_b}_better_than_{config_a}.json'
            
            with open(output_a_path, 'w') as f:
                json.dump(a_better, f, indent=2)
            
            with open(output_b_path, 'w') as f:
                json.dump(b_better, f, indent=2)
            
            print(f"Results saved to {output_a_path} and {output_b_path}")
            
            # Print first example from each comparison for verification
            if a_better:
                first_qid = next(iter(a_better))
                print(f"\nExample where {config_a} succeeded but {config_b} failed:")
                print(json.dumps(a_better[first_qid], indent=2))
            
            if b_better:
                first_qid = next(iter(b_better))
                print(f"\nExample where {config_b} succeeded but {config_a} failed:")
                print(json.dumps(b_better[first_qid], indent=2))
            
    except FileNotFoundError as e:
        print(f"Error: Could not find required JSON files: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
