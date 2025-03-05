import os
import json
import tempfile
import time
import numpy as np
from extract_frames import extract_frames_to_collage
from util import create_question_sentence
from google import genai
from google.genai import types
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
from util import read_json_file, set_environment_variables, save_result
import openai

gemini_safety_settings = [
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
]


def sanitize_message_content(message_content):
    sanitized_content = []
    for item in message_content:
        if isinstance(item, str):
            sanitized_content.append(item)
        elif isinstance(item, list):
            for i in item:
                if isinstance(i, str):
                    sanitized_content.append(i)
                else:
                    sanitized_content.append(str(i))
        else:
            sanitized_content.append(str(item))
    return sanitized_content

def execute_dynamic_sampling_agent(temperature=0):
    max_iterations = int(os.getenv("MAX_ITERATIONS"))

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
    video_duration_seconds = int(video_durations.get(video_id))
    minutes = video_duration_seconds // 60
    seconds = video_duration_seconds % 60
    duration_str = f"{minutes:02d}:{seconds:02d}"
    
    # Helper function to convert timestamp to seconds
    def timestamp_to_seconds(timestamp):
        parts = timestamp.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    # Helper function to convert seconds to timestamp
    def seconds_to_timestamp(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    # UCT parameters
    c = 0.5  # Exploration constant
    decay_threshold = 0.01
    # Controls how quickly relevance decays with distance
    # Effectively split the video into 10 segments
    decay_rate = decay_threshold ** (10/video_duration_seconds)

    print(f"Decay rate: {decay_rate}")
    total_frames = video_duration_seconds
    
    # Initialize UCT tracking variables
    frame_relevance = np.zeros(total_frames)  # G(i) - sum of relevance scores
    frame_visits = np.zeros(total_frames)     # N(i) - number of visits
    
    # Function to update frame relevance with exponential decay
    def update_frame_relevance(timestamp, relevance_score, sample_weight=1.0):
        frame_idx = np.clip(timestamp_to_seconds(timestamp), 0, total_frames - 1)
        frame_relevance[frame_idx] += relevance_score * sample_weight
        frame_visits[frame_idx] += sample_weight
        
        # Apply exponential decay to neighboring frames
        for i in range(total_frames):
            if i != frame_idx:
                distance = abs(i - frame_idx)
                decay_factor = decay_rate ** distance
                # Only update if the decay factor is significant
                if decay_factor > decay_threshold:
                    frame_relevance[i] += relevance_score * decay_factor * sample_weight
                    frame_visits[i] += decay_factor * sample_weight
    
    # Function to calculate UCT score for each frame
    def calculate_uct_scores():
        uct_scores = np.zeros(total_frames)

        print(f"Iteration: {iteration}")
        
        for i in range(total_frames):
            if frame_visits[i] > 0:
                # Exploitation term: average relevance score
                exploitation = frame_relevance[i] / frame_visits[i]
                # Exploration term: UCT formula
                exploration = c * np.sqrt(np.log(iteration + 1) / frame_visits[i])
                # print(f"Frame {i}: Exploitation: {exploitation}, Exploration: {exploration}, UCT score: {exploitation + exploration}")
                uct_scores[i] = exploitation + exploration
            else:
                # For unvisited frames, assign a high exploration value
                uct_scores[i] = c * np.sqrt(np.log(iteration + 1))
        
        # print(f"UCT scores: {uct_scores}")
        # print(f"Frame visits: {frame_visits}")
        # print(f"Frame relevance: {frame_relevance}")

        return uct_scores
    
    # Function to select the next frame to sample based on UCT scores
    def select_next_frame():
        uct_scores = calculate_uct_scores()
        frame_idx = np.argmax(uct_scores)
        timestamp = seconds_to_timestamp(frame_idx * video_duration_seconds // total_frames)
        print(f"Sampled frame at {timestamp}")
        return frame_idx

    frame_relevance_and_decision_schema = {
        "type": "object",
        "properties": {
            "relevance_score": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "relevance": {"type": "number"}
                },
                "required": ["explanation", "relevance"],
                "propertyOrdering": ["explanation", "relevance"],
                "additionalProperties": False
            },
            "reasoning": {"type": "string"},
            "decision": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["answer", "sample_more"]},
                    "answer": {"type": "string", "enum": ["Option A", "Option B", "Option C", "Option D", "Option E"]},
                    "confidence": {"type": "number"}
                },
                "required": ["type", "answer", "confidence"],
                "propertyOrdering": ["type", "answer", "confidence"],
                "additionalProperties": False
            }
        },
        "required": ["relevance_score", "reasoning", "decision"],
        "propertyOrdering": ["relevance_score", "reasoning", "decision"],
        "additionalProperties": False
    }
    
    final_answer_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "answer": {"type": "string", "enum": ["Option A", "Option B", "Option C", "Option D", "Option E"]}
        },
        "required": ["reasoning", "answer"],
        "propertyOrdering": ["reasoning", "answer"],
        "additionalProperties": False
    }
    
    # Initial prompt
    system_prompt_role = "You are a video analysis expert. Your task is to answer a question about a first-person video based on the frames provided. In the question, C refers to the camera wearer." \
         if os.getenv("DATASET") == "egoschema" else \
        "You are a video analysis expert. Your task is to analyze a video and provide a detailed description of the visual content. "
    system_prompt = (
        system_prompt_role +
        "Think step by step and analyze the visual content carefully. "
        f"You will be shown frames from a video that lasts from 00:00 to {duration_str}. "
        "For each set of frames, you'll be asked to rate their relevance to answering the question on a scale from 0 to 1. "
        "A score of 1 means the frame is highly relevant to answering the question, while 0 means it's irrelevant. "
        "After seeing several frames, you'll decide whether to sample more frames or provide your final answer. "
        "If you have enough information to answer, provide your final answer with justification."
    )
    
    # Get model name from environment or use default
    model_name = os.getenv("MODEL", "gemini-2.0-flash")
    
    # Dynamic sampling loop
    prediction_result = -1
    
    # Initialize appropriate client based on model
    if "gemini" in model_name:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    else:  # OpenAI models
        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

    def format_message_for_openai(message_content):
        formatted_messages = [{"role": "developer", "content": system_prompt}]
        for item in message_content:
            if isinstance(item, str) and item.startswith("{"):
                # This is a JSON response from the model
                formatted_messages.append({"role": "assistant", "content": item})
            elif isinstance(item, str):
                formatted_messages.append({"role": "user", "content": item})
            elif isinstance(item, list) and len(item) > 0:
                content_parts = []
                for part in item:
                    if isinstance(part, str):
                        content_parts.append({"type": "text", "text": part})
                    elif hasattr(part, 'mode') and part.mode == 'RGB':  # PIL Image
                        # Convert PIL image to base64
                        import base64
                        import io
                        buffered = io.BytesIO()
                        part.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}", "detail": "high"}
                        })
                if content_parts:
                    formatted_messages.append({"role": "user", "content": content_parts})
        
        return formatted_messages

    def call_llm(response_schema_name, response_schema):
        if "gemini" in model_name:
            response = client.models.generate_content(
                model=model_name,
                contents=message_content,
                config=types.GenerateContentConfig(
                    max_output_tokens=3000,
                    temperature=temperature,
                    seed=42,
                    system_instruction=system_prompt,
                    safety_settings=gemini_safety_settings,
                    response_mime_type='application/json',
                    response_schema=response_schema
                )
            ).text
        else:  # OpenAI models
            formatted_messages = format_message_for_openai(message_content)
            response = client.chat.completions.create(
                model=model_name,
                messages=formatted_messages,
                temperature=temperature,
                seed=42,
                response_format={ "type": "json_schema", "json_schema": {"name": response_schema_name,
                                                                         "strict": True, "schema": response_schema}},
                max_tokens=3000,
            )
            print(f"Prompt tokens: {response.usage.prompt_tokens}")
            print(f'Cached prompt tokens: {response.usage.prompt_tokens_details.cached_tokens}')
            print(f"Completion tokens: {response.usage.completion_tokens}")
            print(f"Total tokens: {response.usage.total_tokens}")
            
            response = response.choices[0].message.content
        
        message_content.append(response)
        return json.loads(response)

    def answer_with_options(message_content):
        prompt = f'{question_sentence}\n\nThink step by step and answer the question with one of the options (A, B, C, D, or E).'
        message_content.append(prompt)
        return call_llm("final_answer", final_answer_schema)

    def get_frame_relevance_and_decision(message_content):
        """Ask the model to rate the relevance of each frame and decide whether to sample more frames or answer"""
        prompt = f"Please rate the relevance of each frame to answering the question on a scale from 0 to 1.\n"
        prompt += f"A score of 1 means the frame is highly relevant, while 0 means it's irrelevant.\n"
        prompt += f"For each frame, provide a brief explanation of your rating.\n\n"
        prompt += "After rating the frames and considering all the information obtained so far, decide whether you can answer the question or need to sample more frames.\n"
        prompt += "If you choose to answer, provide your answer and confidence level (0-1)."

        message_content.append(prompt)
        return call_llm("frame_relevance_and_decision", frame_relevance_and_decision_schema)

    message_content = []
    
    for iteration in range(max_iterations):
        # Select the next frame to sample using UCT
        next_frame = select_next_frame()
        
        # Extract the selected frame
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as frame_file:
            frame_path = frame_file.name
            timestamp = seconds_to_timestamp(next_frame * video_duration_seconds // total_frames)
            success = extract_frames_to_collage(
                video_path=video_path,
                output_path=frame_path,
                start_time=timestamp,
                end_time=timestamp,
                num_frames=1,
                grid_size=(1, 1),
                output_size=768 if "gemini" in model_name else 512
            )
            
            if not success:
                print(f"Warning: Could not extract frame at {timestamp}, trying another frame")
                # Mark this frame as visited to avoid selecting it again
                frame_visits[next_frame] += 0.1
                continue
            
            # Load the frame image
            frame_image = Image.open(frame_path)
            
            # Create prompt for this frame
            user_prompt = [
                f"Here is a frame sampled from the video at {timestamp}",
                frame_image
            ]
            message_content.append(user_prompt)
            
            # Get relevance score and decision for this frame
            decision_response = get_frame_relevance_and_decision(message_content)
            
            # Update UCT values based on relevance score
            relevance = decision_response["relevance_score"]["relevance"]
            update_frame_relevance(timestamp, relevance, sample_weight=1.0)
            
            # Clean up temporary file
            os.remove(frame_path)
        
        # Check the decision from the model
        if decision_response["decision"]["type"] == "answer":
            # Extract the final answer
            answer = decision_response["decision"]["answer"]
            confidence = decision_response["decision"]["confidence"]
            print(f"Final answer: {answer} (confidence: {confidence})")
            # # If confidence is low but we still have iterations left, continue sampling
            # if confidence < 0.7 and iteration < max_iterations - 1:
            #     print(f"Confidence is low ({confidence}), continuing to sample more frames")
            #     continue
            break
    
    # If we've exhausted all iterations without an answer, force a final answer
    if prediction_result == -1:
        response = answer_with_options(message_content)
        answer = response["answer"]
        # Convert answer to prediction result (0-4)
        option_mapping = {"Option A": 0, "Option B": 1, "Option C": 2, "Option D": 3, "Option E": 4}
        prediction_result = option_mapping.get(answer, -1)
    
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
    parser.add_argument('--model', type=str, default="gemini-2.0-flash",
                       help="Model to use: gemini-2.0-flash or gpt-4o")
    parser.add_argument('--max_iterations', type=int, default=10,
                       help="Maximum number of dynamic sampling rounds")
    parser.add_argument('--partition', type=str, default="subset",
                       help="Partition to use: fullset, subset")
    args = parser.parse_args()

    # Set dataset-specific environment variables
    os.environ["DATASET"] = args.dataset
    os.environ["MODEL"] = args.model
    os.environ["MAX_ITERATIONS"] = str(args.max_iterations)

    if args.dataset == "egoschema":
        os.environ["QUESTION_FILE_PATH"] = f"data/egoschema/{args.partition}_uct_{args.model}_max_iter_{args.max_iterations}.json"
        os.environ["VIDEO_DIR_PATH"] = "/simurgh/u/akhatua/VideoMultiAgents/data/egoschema"
        os.environ["VIDEO_DURATIONS"] = "data/egoschema/durations.json"
    elif args.dataset == "nextqa":
        os.environ["QUESTION_FILE_PATH"] = "data/nextqa/val_uct.json"
        os.environ["VIDEO_DIR_PATH"] = "/simurgh/u/akhatua/VideoMultiAgents/data/nextqa/NExTVideo"
        os.environ["VIDEO_DURATIONS"] = "data/nextqa/durations.json"
    elif args.dataset == "momaqa":
        os.environ["QUESTION_FILE_PATH"] = "data/momaqa/test_uct.json"
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
    print(f"Using model: {args.model}")
    
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
