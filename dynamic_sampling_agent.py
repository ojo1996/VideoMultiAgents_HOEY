import os
import json
import tempfile
import time
from extract_frames import extract_frames_to_collage
from util import create_question_sentence
from google import genai
from google.genai import types
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
from util import read_json_file, set_environment_variables, save_result

def sanitize_message_content(message_content):
    sanitized_content = []
    for item in message_content:
        if isinstance(item, str):
            sanitized_content.append(item)
        else:
            sanitized_content.append("<image>")
    return sanitized_content

def execute_dynamic_sampling_agent():
    # Load question data and create question sentence
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    question_sentence = create_question_sentence(target_question_data)
    
    # Get video path
    video_file_name = os.getenv("VIDEO_FILE_NAME")
    video_dir = os.getenv("VIDEO_DIR_PATH")
    video_durations_path = os.getenv("VIDEO_DURATIONS")
    if os.getenv("DATASET") == "nextqa":
        video_path = f'{video_dir}/{video_file_name.replace("-", "/")}.mp4'
    else:
        video_path = f'{video_dir}/{video_file_name}.mp4'
    
    # Get video duration
    with open(video_durations_path, 'r') as f:
        video_durations = json.load(f)
    
    # Get duration for current video
    video_id = video_file_name.split('-')[0] if os.getenv("DATASET") == "nextqa" else video_file_name
    video_duration_seconds = video_durations.get(video_id, 60)  # Default to 60 seconds if not found
    minutes = int(video_duration_seconds // 60)
    seconds = int(video_duration_seconds % 60)
    duration_str = f"{minutes:02d}:{seconds:02d}"
    
    # Configure Gemini API
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_name = 'gemini-2.0-flash'
    
    # Define JSON schemas for responses
    analysis_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "decision": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["answer", "more_frames"]},
                    "answer": {"type": "string"},
                    "start_timestamp": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$", "description": "Timestamp in mm:ss format"},
                    "end_timestamp": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$", "description": "Timestamp in mm:ss format"},
                    "num_frames": {"type": "string", "description": "Number of frames to sample", "enum": ["1", "4", "16"]}
                },
                "required": ["type"],
                "propertyOrdering": ["type", "answer", "start_timestamp", "end_timestamp", "num_frames"]
            }
        },
        "required": ["reasoning", "decision"],
        "propertyOrdering": ["reasoning", "decision"]
    }

    answer_more_frames_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "decision": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["answer", "more_frames"]},
                    "answer": {"type": "string", "enum": ["Option A", "Option B", "Option C", "Option D", "Option E"]},
                    "start_timestamp": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$", "description": "Timestamp in mm:ss format"},
                    "end_timestamp": {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$", "description": "Timestamp in mm:ss format"},
                    "num_frames": {"type": "string", "description": "Number of frames to sample", "enum": ["1", "4", "16"]}
                },
                "required": ["type"],
                "propertyOrdering": ["type", "answer", "start_timestamp", "end_timestamp", "num_frames"]
            }
        },
        "required": ["reasoning", "decision"],
        "propertyOrdering": ["reasoning", "decision"]
    }
    
    final_answer_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "answer": {"type": "string", "enum": ["Option A", "Option B", "Option C", "Option D", "Option E"]}
        },
        "required": ["reasoning", "answer"],
        "propertyOrdering": ["reasoning", "answer"]
    }
    
    # Initial prompt
    system_prompt = (
        "You are a video analysis expert. Your task is to answer a question about a video based on the frames provided. "
        "Think step by step and analyze the visual content carefully. "
        f"If you are even slightly unsure about the answer, you should request specific segments of the video by providing start and end timestamps in mm:ss format. "
        f"The valid range for timestamps is from 00:00 to {duration_str}. "
        "You also need to specify how many frames to sample to trade off detail with context. "
        "If you want to zoom in on a few frames to get more details choose a smaller number of frames. "
        "Otherwise, choose a larger number of frames to get a broader context. "
        "The valid number of frames to sample are 1, 4, or 16. "
        "If you have enough information to answer, provide your final answer with justification."
    )
    
    # Dynamic sampling loop
    max_iterations = 10
    prediction_result = -1

    def answer_with_options(message_content, allow_sample=False):
        prompt = f"{question_sentence}\n\nThink step by step and answer the question with one of the options (A, B, C, D, or E)."
        if allow_sample:
            prompt += " If you are even slightly unsure about any of the options, you should sample more frames."
        message_content.append(prompt)
        # Force a final answer on the last iteration
        response = client.models.generate_content(
            model=model_name,
            contents=message_content,
            config=types.GenerateContentConfig(
                max_output_tokens=3000,
                temperature=0.7,
                seed=42,
                system_instruction=system_prompt,
                safety_settings= [
                    types.SafetySetting(
                        category='HARM_CATEGORY_HATE_SPEECH',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_HARASSMENT',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_DANGEROUS_CONTENT',
                        threshold='BLOCK_NONE'
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_CIVIC_INTEGRITY',
                        threshold='BLOCK_NONE'
                    ),
                ],
                response_mime_type='application/json',
                response_schema=answer_more_frames_schema if allow_sample else final_answer_schema
            )
        )
        return response

    def sample_more_frames(response_json):
        # Extract timestamps for the next set of frames
        start_timestamp = response_json["decision"]["start_timestamp"]
        end_timestamp = response_json["decision"]["end_timestamp"]
        
        # Get number of frames to sample (default to 16 if not specified)
        num_frames = 16
        grid_size = (4, 4)
        if "num_frames" in response_json["decision"]:
            num_frames_str = response_json["decision"]["num_frames"]
            num_frames = int(num_frames_str)
            if num_frames == 1:
                grid_size = (1, 1)
            elif num_frames == 4:
                grid_size = (2, 2)
            elif num_frames == 16:
                grid_size = (4, 4)
        
        print(f"Model requested {num_frames} frames from {start_timestamp} to {end_timestamp}")
        
        # Extract new frames based on the requested timestamps
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            collage_path = temp_file.name
            success = extract_frames_to_collage(
                video_path=video_path,
                output_path=collage_path,
                start_time=start_timestamp,
                end_time=end_timestamp,
                num_frames=num_frames,
                grid_size=grid_size,
                output_size=768
            )
            if not success:
                error_message = f"Error: Iteration {iteration + 1} could not extract frames from {start_timestamp} to {end_timestamp} for video {video_path}"
                print(error_message)
                return -1, message_content + [error_message]
            
            # Load the new collage image
            collage_image = Image.open(collage_path)
            
            # Update the prompt for the next iteration
            user_prompt = [
                f"Here are the {num_frames} frames from {start_timestamp} to {end_timestamp} that you requested. Please continue your analysis.",
                collage_image
            ]
            message_content.append(user_prompt)
    
    for iteration in range(max_iterations):
        # For the first iteration, sample frames uniformly from the entire video
        if iteration == 0:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                collage_path = temp_file.name
                success = extract_frames_to_collage(
                    video_path=video_path,
                    output_path=collage_path,
                    num_frames=16,
                    grid_size=(4, 4),
                    output_size=768
                )
                if not success:
                    error_message = f"Error: Iteration {iteration + 1} could not extract frames from 00:00 to {duration_str} for video {video_path}"
                    print(error_message)
                    return -1, [error_message]
                
                # Load the collage image
                collage_image = Image.open(collage_path)
                
                # Create the prompt for the first iteration
                user_prompt = [
                    f'{target_question_data["question"]}\n\nThese are uniformly sampled frames from the 00:00 to {duration_str}. Analyze them carefully.',
                    collage_image
                ]
                message_content = user_prompt

        # Generate response with appropriate schema
        if iteration < max_iterations - 1:
            response = client.models.generate_content(
                model=model_name,
                contents=message_content,
                config=types.GenerateContentConfig(
                    max_output_tokens=3000,
                    temperature=0.7,
                    seed=42,
                    system_instruction=system_prompt,
                    safety_settings= [
                        types.SafetySetting(
                            category='HARM_CATEGORY_HATE_SPEECH',
                            threshold='BLOCK_NONE'
                        ),
                        types.SafetySetting(
                            category='HARM_CATEGORY_HARASSMENT',
                            threshold='BLOCK_NONE'
                        ),
                        types.SafetySetting(
                            category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                            threshold='BLOCK_NONE'
                        ),
                        types.SafetySetting(
                            category='HARM_CATEGORY_DANGEROUS_CONTENT',
                            threshold='BLOCK_NONE'
                        ),
                        types.SafetySetting(
                            category='HARM_CATEGORY_CIVIC_INTEGRITY',
                            threshold='BLOCK_NONE'
                        ),
                    ],
                    response_mime_type='application/json',
                    response_schema=analysis_schema
                )
            )
        else:
            response = answer_with_options(message_content, allow_sample=False)
        
        # Process the response
        response_json = json.loads(response.text)

        message_content.append(response.text)
        
        # Print the reasoning
        print(f"Iteration {iteration + 1} reasoning:")
        print(response_json["reasoning"])
        
        # Process based on response type
        if iteration < max_iterations - 1 and "decision" in response_json and response_json["decision"]["type"] == "more_frames":
            sample_more_frames(response_json)
        else:
            # Extract the final answer
            if iteration < max_iterations - 1:
                answer = response_json["decision"]["answer"]
                response = answer_with_options(message_content, allow_sample=True)
                response_json = json.loads(response.text)
                message_content.append(response.text)
                if response_json["decision"]["type"] == "answer":
                    answer = response_json["decision"]["answer"]
                else:
                    sample_more_frames(response_json)
                    continue
            else:
                answer = response_json["answer"]
            
            print(f"Final answer: {answer}")
            
            # Convert answer to prediction result (0-4)
            option_mapping = {"Option A": 0, "Option B": 1, "Option C": 2, "Option D": 3, "Option E": 4}
            prediction_result = option_mapping.get(answer, -1)
            
            # Clean up temporary files
            if os.path.exists(collage_path):
                os.remove(collage_path)
            
            break
    
    # Print the final result
    if os.getenv("DATASET") in ["egoschema", "nextqa"]:
        if 0 <= prediction_result <= 4:
            print(
                f"Truth: {target_question_data['truth']}, "
                f"Pred: {prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})"
            )
        else:
            print("Error: Invalid prediction result value")
    elif os.getenv("DATASET") == "momaqa":
        print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result}")
    print("******************************************************")

    return prediction_result, message_content

def process_single_video(dataset, video_data):
    """
    Process a single video with dynamic sampling agent.
    
    Args:
        dataset: Name of the dataset being processed
        video_data: Tuple of (video_id, json_data)
    """
    video_id, json_data = video_data
    try:
        print(f"Processing video_id: {video_id}")
        print(f"JSON data: {json_data}")

        # Set environment variables for this process
        set_environment_variables(dataset, video_id, json_data)

        # Execute dynamic sampling agent
        pred, message_content = execute_dynamic_sampling_agent()
        
        # Save results
        print(f"Results for video {video_id}: {pred}")
        save_result(os.getenv("QUESTION_FILE_PATH"), video_id, "prompt", 
                   sanitize_message_content(message_content), pred, save_backup=False)
        
        return True
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        import traceback
        print(traceback.format_exc())
        time.sleep(1)
        return False

def get_unprocessed_videos(question_file_path, max_items=6000):
    """
    Get a list of all unprocessed videos from the question file.
    
    Args:
        question_file_path: Path to the JSON file containing video questions
    
    Returns:
        List of tuples containing (video_id, json_data) for unprocessed videos
    """
    dict_data = read_json_file(question_file_path)
    unprocessed_videos = []
    for i, (video_id, json_data) in enumerate(list(dict_data.items())[:max_items]):
        if "pred" not in json_data.keys() or json_data["pred"] == -2:
            unprocessed_videos.append((video_id, json_data))
    return unprocessed_videos

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run dynamic sampling agent on video datasets")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to use: egoschema, nextqa, etc.")
    parser.add_argument('--num_workers', type=int, default=None, 
                       help="Number of worker processes. Defaults to CPU count - 1")
    args = parser.parse_args()

    # Set dataset-specific environment variables
    os.environ["DATASET"] = args.dataset
    if args.dataset == "egoschema":
        os.environ["QUESTION_FILE_PATH"] = "data/egoschema/subset_dynamic_sampling.json"
        os.environ["VIDEO_DIR_PATH"] = "/simurgh/u/akhatua/VideoMultiAgents/data/egoschema"
        os.environ["VIDEO_DURATIONS"] = "data/egoschema/durations.json"
    elif args.dataset == "nextqa":
        os.environ["QUESTION_FILE_PATH"] = "data/nextqa/val_dynamic_sampling.json"
        os.environ["VIDEO_DIR_PATH"] = "/simurgh/u/akhatua/VideoMultiAgents/data/nextqa/NExTVideo"
        os.environ["VIDEO_DURATIONS"] = "data/nextqa/durations.json"
    elif args.dataset == "momaqa":
        os.environ["QUESTION_FILE_PATH"] = "data/momaqa/test_dynamic_sampling.json"
        os.environ["VIDEO_DIR_PATH"] = "/root/nas_momaqa/videos"
        os.environ["VIDEO_DURATIONS"] = "data/momaqa/durations.json"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Get list of unprocessed videos
    unprocessed_videos = get_unprocessed_videos(os.getenv("QUESTION_FILE_PATH"))
    
    # Determine number of worker processes
    num_workers = args.num_workers if args.num_workers else max(1, cpu_count() - 1)

    print(f"Starting processing with {num_workers} workers")
    print(f"Found {len(unprocessed_videos)} unprocessed videos")
    
    # Create process pool and process videos in parallel
    with Pool(num_workers) as pool:
        # Create a partial function with fixed arguments
        process_func = partial(process_single_video, args.dataset)
        
        # Process videos in parallel and collect results
        results = pool.map(process_func, unprocessed_videos)
        
        # Print summary
        successful = sum(1 for r in results if r)
        failed = len(results) - successful
        print(f"\nProcessing complete:")
        print(f"Successfully processed: {successful} videos")
        print(f"Failed to process: {failed} videos")

if __name__ == "__main__":
    main()
