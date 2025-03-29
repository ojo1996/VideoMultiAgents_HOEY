import json
import os
import logging
import base64
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import cv2
from difflib import SequenceMatcher
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Cost constants
IMAGE_PROCESSING_COST = 0.001275  # Cost per image in low resolution mode
INPUT_TOKEN_COST = 0.15 / 1_000_000  # Cost per input token
OUTPUT_TOKEN_COST = 0.60 / 1_000_000  # Cost per output token

class MetricsTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.total_time = 0.0
        self.total_images = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.processed_videos = 0
        
    def add_api_call(self, completion):
        """Track costs from an API call"""
        self.total_input_tokens += completion.usage.prompt_tokens
        self.total_output_tokens += completion.usage.completion_tokens
        input_cost = completion.usage.prompt_tokens * INPUT_TOKEN_COST
        output_cost = completion.usage.completion_tokens * OUTPUT_TOKEN_COST
        self.total_cost += input_cost + output_cost
        
    def add_image_processing(self):
        """Track costs from image processing"""
        self.total_images += 1
        self.total_cost += IMAGE_PROCESSING_COST
        
    def log_metrics(self):
        """Log current metrics"""
        logging.info(f"""
Current Processing Metrics:
--------------------------
Processed Videos: {self.processed_videos}
Total Images Processed: {self.total_images}
Total Input Tokens: {self.total_input_tokens}
Total Output Tokens: {self.total_output_tokens}
Total Cost: ${self.total_cost:.2f}
Total Time: {self.total_time:.2f} seconds
Average Cost per Video: ${(self.total_cost / max(1, self.processed_videos)):.2f}
Average Time per Video: {(self.total_time / max(1, self.processed_videos)):.2f} seconds
""")

# Initialize global metrics tracker
metrics = MetricsTracker()

def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/vlm_caption_processing_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )

def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_frame(video_path: str, frame_number: int) -> str:
    """Extract a frame from video and save it temporarily"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not extract frame {frame_number} from video")
    
    # Save frame temporarily
    temp_frame_path = f"temp_frame_{frame_number}.jpg"
    cv2.imwrite(temp_frame_path, frame)
    return temp_frame_path

def calculate_caption_similarity(caption1: str, caption2: str) -> float:
    """Calculate similarity between two captions using sequence matcher"""
    return SequenceMatcher(None, caption1, caption2).ratio()

def find_caption_chunks(captions: List[str], similarity_threshold: float = 0.6, min_chunk_size: int = 5) -> List[Tuple[int, int, str]]:
    """
    Find chunks of similar captions and return (start_idx, end_idx, representative_caption)
    Args:
        captions: List of captions
        similarity_threshold: threshold above which captions are considered similar (0.0 to 1.0)
        min_chunk_size: minimum number of frames in a chunk
    """
    chunks = []
    current_chunk_start = 0
    current_base_caption = captions[0]
    chunk_captions = [current_base_caption]
    
    for i in range(1, len(captions)):
        # Calculate average similarity with all captions in current chunk
        avg_similarity = sum(calculate_caption_similarity(cap, captions[i]) 
                           for cap in chunk_captions) / len(chunk_captions)
        
        # Start new chunk if caption is significantly different AND current chunk is big enough
        current_chunk_size = i - current_chunk_start
        if avg_similarity < similarity_threshold and current_chunk_size >= min_chunk_size:
            # Add current chunk
            chunks.append((current_chunk_start, i-1, current_base_caption))
            # Start new chunk
            current_chunk_start = i
            current_base_caption = captions[i]
            chunk_captions = [current_base_caption]
        else:
            # Add to current chunk
            chunk_captions.append(captions[i])
    
    # Add final chunk
    if len(captions) - current_chunk_start < min_chunk_size and chunks:
        # If final chunk is too small, merge with previous chunk
        last_start, _, last_caption = chunks.pop()
        chunks.append((last_start, len(captions)-1, last_caption))
    else:
        chunks.append((current_chunk_start, len(captions)-1, current_base_caption))
    
    # Log chunk information
    logging.info(f"Split {len(captions)} captions into {len(chunks)} chunks")
    for start, end, caption in chunks:
        chunk_size = end - start + 1
        logging.info(f"Chunk {start}-{end} ({chunk_size} frames): {caption}")
        if chunk_size < min_chunk_size:
            logging.warning(f"Chunk size {chunk_size} is less than minimum {min_chunk_size}")
    
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
def generate_scene_graph_and_caption(
    frame_path: str,
    original_caption: str,
    yolo_detections: List[Dict],
    client: OpenAI,
    previous_scene_graph: Optional[List[List[str]]] = None
) -> Tuple[List[List[str]], str]:
    """Generate scene graph and enriched caption using GPT-4-Vision"""
    
    # Track image processing cost
    metrics.add_image_processing()
    
    # Encode image
    base64_image = encode_image(frame_path)
    
    # Format YOLO detections
    detection_text = "\n".join([
        f"- {d['class_name']} (confidence: {d['confidence']:.2f}) at coordinates: {d['coordinates']}"
        for d in yolo_detections
    ])
    
    # Include previous scene graph context if available
    previous_context = ""
    if previous_scene_graph:
        previous_context = "\nPrevious frame's scene graph for consistency:\n"
        for triplet in previous_scene_graph:
            previous_context += f"[{triplet[0]}, {triplet[1]}, {triplet[2]}]\n"
    
    prompt = f"""Analyze this video frame using the following components:

1. Original Caption: {original_caption}
2. Detected Objects (YOLO):
{detection_text}
3. Previous Context:{previous_context if previous_context else " None"}

Generate two distinct outputs:

1. SCENE GRAPH:
Generate a list of triplets [subject, relation, object] that describe the scene.
Each triplet MUST have exactly three elements:
- subject: the entity performing the action or being described
- relation: the action or relationship
- object: the target entity or state

Format each triplet as: [subject, relation, object]

Example triplets:
[person, sitting_at, desk]
[laptop, displaying, content]
[cup, next_to, keyboard]
[person, holding, mouse]

Requirements:
- Each triplet must be a complete [subject, relation, object]
- Use simple present tense for relations
- No helper verbs (use 'holds' not 'is_holding')
- Only use observed entities as subjects and objects
- Relations should be actions or spatial relationships

Output format:
<scene_graph>
[subject1, relation1, object1]
[subject2, relation2, object2]
...
</scene_graph>

2. ENRICHED CAPTION:
Create a natural language description that:
- MUST preserve all original caption actions/objects
- Incorporates scene graph relationships
- Uses temporal connectors (while, as, where)
- Stays factual without assumptions

Output format:
<enriched_caption>caption text</enriched_caption>"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise scene graph generator. Generate triplets in exact [subject, relation, object] format."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Track API call metrics
        metrics.add_api_call(response)
        
        result = response.choices[0].message.content
        
        # Log the response
        logging.info("Model response:")
        logging.info("-" * 80)
        logging.info(result)
        logging.info("-" * 80)
        
        # Extract scene graph and convert to list of triplets
        scene_graph_text = result[result.find("<scene_graph>")+13:result.find("</scene_graph>")]
        # Parse triplets ensuring they're in [subject, relation, object] format
        scene_graph_triplets = []
        for line in scene_graph_text.strip().split('\n'):
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                # Remove brackets and split by comma
                parts = [p.strip() for p in line[1:-1].split(',')]
                if len(parts) == 3:  # Only accept valid triplets
                    scene_graph_triplets.append([p.strip(' "\'') for p in parts])
        
        # Extract enriched caption
        enriched_caption = result[result.find("<enriched_caption>")+18:result.find("</enriched_caption>")]
        
        return scene_graph_triplets, enriched_caption.strip()
        
    except Exception as e:
        logging.error(f"Error generating scene graph and caption: {str(e)}")
        raise
    finally:
        # Clean up temporary frame file
        if os.path.exists(frame_path):
            os.remove(frame_path)

def process_chunk(
    chunk_data: Tuple[int, Tuple[int, int, str], str, List[Dict], OpenAI, List[List[str]], str],
) -> Dict:
    """Process a single chunk of video in parallel"""
    chunk_idx, (start_frame, end_frame, base_caption), video_path, unique_detections, client, previous_scene_graph, video_id = chunk_data
    
    try:
        logging.info(f"Processing chunk {chunk_idx} (frames {start_frame}-{end_frame}) for video {video_id}")
        
        # Get middle frame for visual reference
        mid_frame = (start_frame + end_frame) // 2
        
        # Extract representative frame from video
        frame_path = extract_frame(video_path, mid_frame)
        
        # Generate scene graph and enriched caption
        scene_graph_triplets, enriched_caption = generate_scene_graph_and_caption(
            frame_path,
            base_caption,
            unique_detections,
            client,
            previous_scene_graph
        )
        
        return {
            'chunk_idx': chunk_idx,
            'time_start': start_frame,
            'time_end': end_frame,
            'original_captions': [base_caption],  # Will be updated with full list later
            'scene_graph': scene_graph_triplets,
            'enriched_caption': enriched_caption,
            'yolo_detections': unique_detections
        }
        
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_idx}: {str(e)}")
        raise

def process_video(
    video_path: str,
    video_id: str,
    captions: List[str],
    yolo_data: Dict,
    client: OpenAI,
    output_path: str,
    max_workers: int = 8
):
    """Process a single video with parallel chunk processing"""
    start_time = time.time()
    try:
        # Get YOLO detections for this video
        video_yolo_data = yolo_data.get(video_id, {})
        if not video_yolo_data:
            logging.warning(f"No YOLO detections found for video {video_id}")
        
        # Find chunks of similar captions
        caption_chunks = find_caption_chunks(captions)
        
        # Prepare chunk data for parallel processing
        chunk_data = []
        previous_scene_graph = None
        
        for chunk_idx, (start_frame, end_frame, base_caption) in enumerate(caption_chunks):
            # Collect all YOLO detections for this chunk
            chunk_detections = []
            for frame_idx in range(start_frame, end_frame + 1):
                frame_key = f"{frame_idx:04d}.jpg"
                frame_detections = video_yolo_data.get(frame_key, [])
                chunk_detections.extend(frame_detections)
            
            # Remove duplicate detections by class_name
            seen_classes = set()
            unique_detections = []
            for det in chunk_detections:
                if det['class_name'] not in seen_classes:
                    seen_classes.add(det['class_name'])
                    unique_detections.append(det)
            
            # Prepare data for this chunk
            chunk_data.append((
                chunk_idx,
                (start_frame, end_frame, base_caption),
                video_path,
                unique_detections,
                client,
                previous_scene_graph,
                video_id
            ))
        
        # Process chunks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, data): data
                for data in chunk_data
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future][0]
                try:
                    chunk_result = future.result()
                    # Update with full list of original captions
                    start_frame = chunk_result['time_start']
                    end_frame = chunk_result['time_end']
                    chunk_result['original_captions'] = captions[start_frame:end_frame + 1]
                    results.append(chunk_result)
                    logging.info(f"Completed chunk {chunk_idx} for video {video_id}")
                    
                    # Update previous scene graph for next chunks
                    if chunk_idx < len(caption_chunks) - 1:
                        next_chunk_data = chunk_data[chunk_idx + 1]
                        next_chunk_data = list(next_chunk_data)
                        next_chunk_data[5] = chunk_result['scene_graph']
                        chunk_data[chunk_idx + 1] = tuple(next_chunk_data)
                    
                except Exception as e:
                    logging.error(f"Chunk {chunk_idx} generated an exception: {str(e)}")
                    raise
        
        # Sort results by chunk_idx
        results.sort(key=lambda x: x['chunk_idx'])
        
        # Save all results
        save_results(output_path, video_id, results)
        
        # Update metrics
        metrics.processed_videos += 1
        metrics.total_time += time.time() - start_time
        metrics.log_metrics()
        
        logging.info(f"Completed processing video {video_id}")
        
    except Exception as e:
        logging.error(f"Error processing video {video_id}: {str(e)}")
        raise

def save_results(output_path: str, video_id: str, results: List[Dict]):
    """Save results for a video"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create or load existing results
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse existing file {output_path}, creating new")
                    data = {}
        else:
            data = {}
        
        # Update results for this video
        data[video_id] = results
        
        # Save to file with proper JSON serialization
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"Saved results for video {video_id} to {output_path}")
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

def main():
    setup_logging()
    logging.info("Starting VLM caption processing pipeline")
    
    try:
        # File paths
        base_path = os.path.abspath("/simurgh/u/akhatua/VideoMultiAgents/data")
        videos_path = os.path.join(base_path, "egoschema")
        captions_path = os.path.join(base_path, "egoschema_captions.json")
        yolo_path = os.path.join(base_path, "egoschema_yolo.json")
        output_path = os.path.join(base_path, "egoschema_vlm_captions_test.json")  # Different output file for test run
        
        # Load data
        logging.info("Loading caption data...")
        with open(captions_path, 'r') as f:
            all_captions_data = json.load(f)
        
        logging.info("Loading YOLO detection data...")
        with open(yolo_path, 'r') as f:
            yolo_data = json.load(f)
            
        # Select only first 10 videos for estimation
        captions_data = dict(list(all_captions_data.items())[:10]) # TODO: Remove this 10
        total_dataset_videos = len(all_captions_data)  # Store total number of videos
        logging.info(f"Selected first 10 videos for estimation (out of {total_dataset_videos} total videos)")
        
        # Initialize OpenAI client
        client = OpenAI()
        
        # Process each video
        total_videos = len(captions_data)
        for idx, (video_id, captions) in enumerate(captions_data.items(), 1):
            try:
                logging.info(f"Processing video {idx}/{total_videos}: {video_id}")
                
                # Check if already processed
                if os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        try:
                            existing_data = json.load(f)
                            if video_id in existing_data:
                                logging.info(f"Skipping video {video_id} - already processed")
                                continue
                        except json.JSONDecodeError:
                            pass
                
                # Check if we have YOLO data for this video
                if video_id not in yolo_data:
                    logging.warning(f"Skipping video {video_id} - no YOLO data available")
                    continue
                
                # Process video
                video_path = os.path.join(videos_path, f"{video_id}.mp4")
                if not os.path.exists(video_path):
                    logging.warning(f"Video file not found: {video_path}")
                    continue
                    
                process_video(
                    video_path,
                    video_id,
                    captions,
                    yolo_data,
                    client,
                    output_path,
                    max_workers=8  # Explicitly set to 8 workers
                )
                
            except Exception as e:
                logging.error(f"Error processing video {video_id}: {str(e)}")
                continue
        
        # Final metrics report
        logging.info("\nFinal Metrics Report:")
        logging.info("===================")
        metrics.log_metrics()
        
        # Extrapolate to full dataset using total_dataset_videos
        estimated_total_cost = metrics.total_cost * (total_dataset_videos / metrics.processed_videos)
        estimated_total_time = metrics.total_time * (total_dataset_videos / metrics.processed_videos)
        
        logging.info(f"""
Extrapolated Estimates for Full Dataset ({total_dataset_videos} videos):
---------------------------------------------------------------------
Estimated Total Cost: ${estimated_total_cost:.2f}
Estimated Total Time: {estimated_total_time/3600:.1f} hours
""")
        
        logging.info("Processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 