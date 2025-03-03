import json
import os
import logging
import time
import shutil
from typing import Dict, List
from openai import OpenAI
from collections import defaultdict
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import weave

def validate_environment():
    """Validate that all required environment variables are set"""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

def validate_json_data(data: Dict, data_type: str) -> bool:
    """Validate the structure of loaded JSON data"""
    if not isinstance(data, dict):
        raise ValueError(f"{data_type} data must be a dictionary")
    
    if not data:
        raise ValueError(f"{data_type} data is empty")
    
    if data_type == "captions":
        # Validate caption data structure
        for video_id, captions in data.items():
            if not isinstance(captions, list):
                raise ValueError(f"Captions for video {video_id} must be a list")
            if not all(isinstance(c, str) for c in captions):
                raise ValueError(f"All captions for video {video_id} must be strings")
    
    elif data_type == "yolo":
        # Validate YOLO data structure
        for video_id, frames in data.items():
            if not isinstance(frames, dict):
                raise ValueError(f"YOLO data for video {video_id} must be a dictionary")
            for frame_key, detections in frames.items():
                if not isinstance(detections, list):
                    raise ValueError(f"Detections for frame {frame_key} must be a list")
                for detection in detections:
                    required_keys = {'class', 'class_name', 'confidence', 'coordinates'}
                    if not all(key in detection for key in required_keys):
                        raise ValueError(f"Detection missing required keys: {required_keys}")
    
    return True

# Set up logging
def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/caption_processing_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )

def load_json_file(file_path: str, data_type: str = None) -> Dict:
    logging.info(f"Loading JSON file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if data_type:
            validate_json_data(data, data_type)
            
        logging.info(f"Successfully loaded {file_path} with {len(data)} entries")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        raise

def backup_file(file_path: str):
    """Create a backup of the file if it exists"""
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{file_path}.{timestamp}.backup"
        shutil.copy2(file_path, backup_path)
        logging.info(f"Created backup at {backup_path}")

def save_video_result(output_path: str, video_id: str, video_data: List[Dict]):
    """Append a single video's results to the output file"""
    try:
        # Create the file with empty dict if it doesn't exist
        if not os.path.exists(output_path):
            with open(output_path, 'w') as f:
                json.dump({}, f, indent=2)
        
        # Read current content
        with open(output_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Corrupted output file {output_path}, creating new")
                data = {}
        
        # Update with new video data
        data[video_id] = video_data
        
        # Write back to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"Successfully saved results for video {video_id}")
    except Exception as e:
        logging.error(f"Error saving results for video {video_id}: {str(e)}")
        raise

def validate_frame_key(frame_key: str) -> bool:
    """Validate that frame key matches expected format (####.jpg)"""
    import re
    return bool(re.match(r'^\d{4}\.jpg$', frame_key))

def chunk_video_data(captions: List[str], yolo_detections: Dict, chunk_size: int = 10) -> List[Dict]:
    logging.info(f"Chunking video data: {len(captions)} captions into {chunk_size}-second segments")
    chunks = []
    
    for i in range(0, len(captions), chunk_size):
        chunk_captions = captions[i:i + chunk_size]
        
        # Get corresponding YOLO detections for this chunk
        chunk_objects = set()
        for frame_idx in range(i, min(i + chunk_size, len(captions))):
            frame_key = f"{frame_idx:04d}.jpg"
            if not validate_frame_key(frame_key):
                logging.warning(f"Invalid frame key format: {frame_key}")
                continue
                
            if frame_key in yolo_detections:
                for detection in yolo_detections[frame_key]:
                    if detection['confidence'] > 0.6:  # Filter low confidence detections
                        chunk_objects.add(detection['class_name'])
        
        chunk_info = {
            'time_start': i,
            'time_end': min(i + chunk_size, len(captions)),
            'original_captions': chunk_captions,
            'detected_objects': list(chunk_objects)
        }
        chunks.append(chunk_info)
        logging.debug(f"Created chunk {len(chunks)}: {i}s-{chunk_info['time_end']}s with {len(chunk_objects)} unique objects")
    
    logging.info(f"Created {len(chunks)} chunks")
    return chunks

# Rate limit: max 3 retries, with exponential backoff starting at 4 seconds
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
def generate_enhanced_caption(chunk: Dict, client: OpenAI) -> str:
    logging.info(f"Generating enhanced caption for segment {chunk['time_start']}-{chunk['time_end']}s")
    
    
    prompt = f"""Given the following information about a 10-second video segment:

Original captions for each second: {chunk['original_captions']}
Objects detected in the scene: {', '.join(chunk['detected_objects'])}

Please generate a single concise caption that:
1. Summarizes the main actions and events in one sentence
2. Incorporates the detected objects and their graphical and temporal relations

Generate a single caption."""

    # Log the prompt being sent to the model
    logging.info("Sending prompt to model:")
    logging.info("-" * 80)
    logging.info(prompt)
    logging.info("-" * 80)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a video description expert that creates detailed, accurate captions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        enhanced_caption = completion.choices[0].message.content
        
        # Log the model's response
        logging.info("Model response:")
        logging.info("-" * 80)
        logging.info(enhanced_caption)
        logging.info("-" * 80)
        
        # Log additional completion info
        logging.info(f"Model usage - Prompt tokens: {completion.usage.prompt_tokens}, "
                    f"Completion tokens: {completion.usage.completion_tokens}, "
                    f"Total tokens: {completion.usage.total_tokens}")
        
        return enhanced_caption
    except Exception as e:
        logging.error(f"Error generating caption with GPT-4: {str(e)}")
        raise

def main():
    setup_logging()
    logging.info("Starting caption processing pipeline")
    
    try:
        # Validate environment
        validate_environment()
        
        # File paths
        base_path = os.path.abspath("/simurgh/u/akhatua/VideoMultiAgents/data")
        captions_path = os.path.join(base_path, "egoschema_captions.json")
        yolo_path = os.path.join(base_path, "egoschema_yolo.json")
        output_path = os.path.join(base_path, "egoschema_graph_captions.json")
        
        # Load and validate data
        logging.info("Loading input files...")
        captions_data = load_json_file(captions_path, "captions")
        yolo_data = load_json_file(yolo_path, "yolo")
        
        # Initialize OpenAI client
        client = OpenAI()
        logging.info("OpenAI client initialized")
        
        # Process each video
        total_videos = len(captions_data)
        
        for idx, (video_id, captions) in enumerate(captions_data.items(), 1):
            try:
                logging.info(f"Processing video {idx}/{total_videos}: {video_id}")
                
                # Check if video already processed
                if os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        try:
                            existing_data = json.load(f)
                            if video_id in existing_data:
                                logging.info(f"Skipping video {video_id} - already processed")
                                continue
                        except json.JSONDecodeError:
                            logging.warning("Corrupted output file, continuing with processing")
                
                # Get YOLO detections for this video
                video_yolo = yolo_data.get(video_id, {})
                logging.info(f"Found {len(video_yolo)} YOLO detections for video {video_id}")
                
                # Chunk the video data
                chunks = chunk_video_data(captions, video_yolo)
                
                # Generate enhanced captions for each chunk
                video_enhanced_captions = []
                for chunk_idx, chunk in enumerate(chunks, 1):
                    logging.info(f"Processing chunk {chunk_idx}/{len(chunks)} for video {video_id}")
                    enhanced_caption = generate_enhanced_caption(chunk, client)
                    video_enhanced_captions.append({
                        'time_start': chunk['time_start'],
                        'time_end': chunk['time_end'],
                        'enhanced_caption': enhanced_caption,
                        'original_captions': chunk['original_captions'],
                        'detected_objects': chunk['detected_objects']
                    })
                
                # Save results for this video immediately
                save_video_result(output_path, video_id, video_enhanced_captions)
                logging.info(f"Completed processing video {video_id} with {len(video_enhanced_captions)} enhanced captions")
                
            except Exception as e:
                logging.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
                continue  # Continue with next video if one fails
        
        logging.info("All videos processed successfully")
        
    except Exception as e:
        logging.error(f"Error in main processing pipeline: {str(e)}", exc_info=True)
        raise
    
    logging.info("Caption processing pipeline completed successfully")

if __name__ == "__main__":
    main() 