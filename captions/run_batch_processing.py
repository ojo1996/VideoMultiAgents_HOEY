import os
import json
import subprocess
from datetime import datetime
import pandas as pd

def load_validation_videos(val_path: str) -> set:
    """Load video IDs from validation CSV file"""
    val_df = pd.read_csv(val_path)
    return set(val_df['video'].astype(str))

def process_captions_data(captions_data: dict, valid_video_ids: set) -> dict:
    """Process captions data to extract video IDs after '-' and filter by valid IDs"""
    processed_data = {}
    for key, value in captions_data.items():
        # Extract video ID after the '-'
        video_id = key.split('-')[-1]
        if video_id in valid_video_ids:
            processed_data[video_id] = value
    return processed_data

def get_total_videos():
    base_path = os.path.abspath("/simurgh/u/akhatua/VideoMultiAgents/data")
    captions_path = os.path.join(base_path, "nextqa/nextqa_captions_gpt4o.json")
    val_path = os.path.join(base_path, "nextqa/val.csv")
    
    # Load validation video IDs
    valid_video_ids = load_validation_videos(val_path)
    
    # Load and process captions data
    with open(captions_path, 'r') as f:
        all_captions_data = json.load(f)
    
    # Filter captions based on validation videos
    processed_data = process_captions_data(all_captions_data, valid_video_ids)
    return len(processed_data)

def main():
    BATCH_SIZE = 1000
    total_videos = get_total_videos()
    
    # Assert that we have the expected number of videos
    EXPECTED_VIDEOS = 570
    assert total_videos == EXPECTED_VIDEOS, f"Expected {EXPECTED_VIDEOS} videos, but found {total_videos} videos"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Total videos to process: {total_videos}")
    
    # Calculate number of batches
    num_batches = (total_videos + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch in range(num_batches):
        start_idx = batch * BATCH_SIZE
        end_idx = min((batch + 1) * BATCH_SIZE, total_videos)
        output_suffix = f"gpt4o_batch_{start_idx:06d}_{end_idx:06d}"
        
        print(f"\nProcessing batch {batch + 1}/{num_batches}")
        print(f"Videos {start_idx} to {end_idx}")
        print(f"Output suffix: {output_suffix}")
        
        cmd = [
            "python",
            "vlm_captions.py",
            "--start-idx", str(start_idx),
            "--end-idx", str(end_idx),
            "--output-suffix", output_suffix
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully completed batch {batch + 1}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing batch {batch + 1}: {e}")
            # Continue with next batch even if current one fails
            continue

if __name__ == "__main__":
    main() 