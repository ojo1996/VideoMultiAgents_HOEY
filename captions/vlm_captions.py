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

def process_video(
    video_path: str,
    video_id: str,
    captions: List[str],
    yolo_data: Dict,
    client: OpenAI,
    output_path: str
):
    """Process a single video"""
    try:
        results = []
        previous_scene_graph = None
        
        # Get YOLO detections for this video
        video_yolo_data = yolo_data.get(video_id, {})
        if not video_yolo_data:
            logging.warning(f"No YOLO detections found for video {video_id}")
        
        # Find chunks of similar captions
        caption_chunks = find_caption_chunks(captions)
        
        for chunk_idx, (start_frame, end_frame, base_caption) in enumerate(caption_chunks):
            logging.info(f"Processing chunk {chunk_idx+1}/{len(caption_chunks)} (frames {start_frame}-{end_frame})")
            
            # Get middle frame of chunk for visual reference
            mid_frame = (start_frame + end_frame) // 2
            frame_key = f"{mid_frame:04d}.jpg"
            
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
            
            # Log YOLO detections for debugging
            logging.info(f"Found {len(unique_detections)} unique object types in chunk")
            for det in unique_detections:
                logging.debug(f"Detection: {det['class_name']}")
            
            # Extract representative frame from video
            frame_path = extract_frame(video_path, mid_frame)
            
            # Generate scene graph and enriched caption
            scene_graph_triplets, enriched_caption = generate_scene_graph_and_caption(
                frame_path,
                base_caption,  # Use base caption for generation
                unique_detections,
                client,
                previous_scene_graph
            )
            
            # Store all original captions for this chunk
            chunk_captions = captions[start_frame:end_frame + 1]
            
            # Create one result entry for the entire chunk
            results.append({
                'chunk_idx': chunk_idx,
                'time_start': start_frame,
                'time_end': end_frame,
                'original_captions': chunk_captions,
                'scene_graph': scene_graph_triplets,
                'enriched_caption': enriched_caption,
                'yolo_detections': unique_detections
            })
            
            # Update previous scene graph for next chunk
            previous_scene_graph = scene_graph_triplets
            
            # Save results after each chunk
            save_results(output_path, video_id, results)
            
        logging.info(f"Completed processing video {video_id}")
        
    except Exception as e:
        logging.error(f"Error processing video {video_id}: {str(e)}")
        raise

def save_results(output_path: str, video_id: str, results: List[Dict]):
    """Save results for a video"""
    try:
        # Create or load existing results
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
        
        # Update results for this video
        data[video_id] = results
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=lambda x: x if not isinstance(x, list) else x)
            
        logging.info(f"Saved results for video {video_id}")
        
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
        output_path = os.path.join(base_path, "egoschema_vlm_captions.json")
        
        # Load data
        logging.info("Loading caption data...")
        with open(captions_path, 'r') as f:
            captions_data = json.load(f)
        
        logging.info("Loading YOLO detection data...")
        with open(yolo_path, 'r') as f:
            yolo_data = json.load(f)
            
        # Log some statistics about the data
        logging.info(f"Loaded {len(captions_data)} videos with captions")
        logging.info(f"Loaded YOLO detections for {len(yolo_data)} videos")
        
        # Check for mismatches
        missing_yolo = set(captions_data.keys()) - set(yolo_data.keys())
        if missing_yolo:
            logging.warning(f"Missing YOLO data for videos: {missing_yolo}")
            
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
                    output_path
                )
                
            except Exception as e:
                logging.error(f"Error processing video {video_id}: {str(e)}")
                continue
        
        logging.info("Processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 