import os
import json
import tempfile
import time
from extract_frames import extract_frames_to_collage
from util import create_question_sentence
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
from util import read_json_file, set_environment_variables, save_result
import openai


def execute_vca_agent(temperature=0.5):
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
    
    # Helper function to convert timestamp to seconds
    def timestamp_to_seconds(timestamp):
        parts = timestamp.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    # Helper function to convert seconds to timestamp
    def seconds_to_timestamp(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    # VCA parameters
    total_frames = video_duration_seconds
    frames_per_iteration = 7  # Number of frames to sample per iteration
    
    # Initialize the first segment as the entire video
    current_segment_id = '0'

    # Initialize VCA tracking variables
    memory_buffer = []  # Memory buffer M
    candidate_segments = {'0': {'start': 0, 'end': total_frames - 1, "score": 100, "explanation": "Root segment is the starting point of analysis"}}  # Candidate segments set S
    
    # Schema for reward model output
    def reward_model_schema(new_segments):
        allowed_segment_ids = list(new_segments.keys())
        print('Allowed segment ids: ', allowed_segment_ids)
        return {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_id": {"type": "string", "enum": allowed_segment_ids},
                            "explanation": {"type": "string"},
                            "score": {"type": "integer"}
                        },
                        "required": ["segment_id", "explanation", "score"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["segments"],
            "additionalProperties": False
        }
    
    # Schema for exploration agent output
    def exploration_agent_schema():
        # Filter candidate segments to keep only those with sufficient length for expansion
        filtered_segments = {seg_id: seg_data for seg_id, seg_data in candidate_segments.items() 
                            if seg_data['end'] - seg_data['start'] >= frames_per_iteration}
        allowed_segment_ids = list(filtered_segments.keys())
        return {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "decision": {
                    "type": "string",
                    "enum": ["explore", "answer"]
                },
                "segment_id": {
                    "type": ["string", "null"],
                    "enum": allowed_segment_ids,
                    "description": "The segment ID to explore next, or null if answering"
                },
                "answer": {
                    "type": ["string", "null"],
                    "description": "The final answer if decision is 'answer', or null if exploring",
                    "enum": ["Option A", "Option B", "Option C", "Option D", "Option E"]
                },
            },
            "required": ["reasoning", "decision", "segment_id", "answer"],
            "additionalProperties": False
        }
    
    # Function to uniformly sample frames from a segment
    def uniform_sample(segment_id, num_frames):
        assert num_frames == frames_per_iteration
        
        start_frame = candidate_segments[segment_id]['start']
        end_frame = candidate_segments[segment_id]['end']
        segment_length = end_frame - start_frame + 1
        
        if segment_length <= num_frames:
            # If segment is smaller than requested frames, return all frames
            frame_indices = list(range(start_frame, end_frame + 1))
        else:
            # Uniform sampling
            step = segment_length / (num_frames + 1)
            frame_indices = [int(start_frame + i * step) for i in range(1, num_frames + 1)]
            assert len(frame_indices) == num_frames
        
        frames = []
        frame_paths = []
        
        # Extract frames individually instead of as a collage
        for idx in frame_indices:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as frame_file:
                frame_path = frame_file.name
                
                # Convert frame index to timestamp
                timestamp = seconds_to_timestamp(idx)
                
                # Extract individual frame
                success = extract_frames_to_collage(
                    video_path=video_path,
                    output_path=frame_path,
                    start_time=timestamp,
                    end_time=timestamp,
                    num_frames=1,
                    grid_size=(1, 1),  # Individual frame, not a collage
                    output_size=512
                )
                
                if success:
                    frame_image = Image.open(frame_path)
                    frames.append(frame_image)
                    frame_paths.append(frame_path)
                else:
                    print(f"Warning: Could not extract frame at {timestamp}")
        
        if not frames:
            print(f"Warning: Could not extract any frames from segment {segment_id}")
        
        # Create segments from frame indices
        new_segments = {}
        for i in range(len(frame_indices) + 1):
            if i == 0:
                segment_start = start_frame
                segment_end = frame_indices[0]
            elif i == len(frame_indices):
                segment_start = frame_indices[-1]
                segment_end = end_frame
            else:
                segment_start = frame_indices[i-1]
                segment_end = frame_indices[i]
            
            new_segment_id = f'{segment_id}.{i+1}'
            new_segments[new_segment_id] = {'start': segment_start, 'end': segment_end}

        print(f"Segments: {new_segments}")
        print(f"Frame indices: {frame_indices}")
        print(f"Num frames: {num_frames}")
        
        assert len(new_segments) == num_frames + 1
        
        return frames, frame_indices, new_segments, frame_paths
    
    # Function to update memory buffer
    def update_memory(memory, new_frames, segments_with_scores):
        # Add new frames to memory
        memory.extend(new_frames)
        
        # If memory exceeds limit, remove frames with lowest scores
        memory_limit = 8 if os.getenv("DATASET") == "egoschema" else 16
        if len(memory) > memory_limit:
            # TODO: Sort segments by score and keep only the top frames
            memory = memory[-memory_limit:]
        
        print('Memory size: ', len(memory))
        return memory
    
    # Function to call the reward model
    def call_reward_model(frames, frame_indices, new_segments, iteration):
        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Prepare the prompt based on iteration
        if iteration == 0:
            # First round prompt
            prompt = f"""
/* Task Description */
You are acting as a reward model to guide the video question-answering process, with access to a {total_frames}-frame video ({video_duration_seconds} seconds in duration). You are provided with {len(frame_indices)} uniformly sampled frames from the video, at the following frame indices: {frame_indices}, which divide the video into {len(new_segments)} distinct segments.

/* Segment Information */
"""
            for segment_id, segment_data in new_segments.items():
                start_time = seconds_to_timestamp(segment_data['start'])
                end_time = seconds_to_timestamp(segment_data['end'])
                prompt += f"Segment {segment_id}: [{start_time}, {end_time}]\n"
            
            prompt += f"""
/* Reward Instruction */
Your task is to evaluate the relevance of each segment in answering the question below, to assist in identifying the segment(s) that most effectively answer the question.

{question_sentence}

First provide an explanation based on the specific details observed in each sub-segment, then assign a relevance score from 0% to 100%. Focus on each segment's alignment with the question.

Hint: Since only the start and end frames of each segment are provided, first imagine the possible content within each segment based on these frames before making a decision.

In your explanation, describe your reasoning process and any inferred details about the segment's content, using specific observations from the start and end frames as a basis for these inferences.
"""
        else:
            # Subsequent rounds prompt
            prompt = f"""
/* Task Description */
You are acting as a reward model in a multi-round video question-answering process. You have access to a {total_frames}-frame video ({video_duration_seconds} seconds in duration), along with results from a previous round of evaluation. In this round, one specific segment has been further divided to provide more detailed analysis. You are provided with {len(frame_indices)} new sampled frames to assess these sub-segments in relation to the question, at the following frame indices: {frame_indices}.

{question_sentence}

/* Historical Segment Information */
In the last round, the video is divided into {len(candidate_segments)} segments, each segment was evaluated for its relevance to the goal question. Here are the results from all previous rounds:
"""
            print('Candidate segments: ', candidate_segments)
            for segment_id, segment_data in candidate_segments.items():
                score = segment_data["score"]
                start_time = seconds_to_timestamp(segment_data['start'])
                end_time = seconds_to_timestamp(segment_data['end'])
                prompt += f"Segment {segment_id}: [{start_time}, {end_time}] || Relevance Score: {score}%\n"
            
            prompt += f"""
/* Current Segment Information */
In this round, segment {current_segment_id} has been further explored with {len(frame_indices)} new uniformly sampled frames, dividing it into {len(new_segments)} new sub-segments:
"""
            for segment_id, segment_data in new_segments.items():
                start_time = seconds_to_timestamp(segment_data['start'])
                end_time = seconds_to_timestamp(segment_data['end'])
                prompt += f"Segment {segment_id}: [{start_time}, {end_time}]\n"
            
            prompt += f"""
/* Reward Instruction */
Your task is to evaluate these new sub-segments for relevance to the original goal question based on provided frames, to assist in identifying the segment(s) that most effectively answer the question, while considering the context and results from the previous rounds.

First provide an explanation based on the specific details observed in each sub-segment, then assign a relevance score from 0% to 100%. Focus on each segment's alignment with the question, ensuring consistency with the scores and explanations from previous segments.

Hint: Since only the start and end frames of each segment are provided, first imagine the possible content within each segment based on these frames before making a decision.

In your explanation, describe your reasoning process and any inferred details about the segment's content, using specific observations from the start and end frames as a basis for these inferences.
"""
        
        # Prepare message content with frames
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Add frames to the message content
        for i, frame in enumerate(frames):
            # Convert PIL image to base64
            import base64
            import io
            buffered = io.BytesIO()
            frame.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Add frame to message
            msgs[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}", "detail": "high"}
            })
        
        # Call the model
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=msgs,
            temperature=temperature,
            seed=42,
            response_format={"type": "json_schema", "json_schema": { "name": "reward_model", "schema": reward_model_schema(new_segments), "strict": True}},
            max_tokens=3000,
        )
        
        # Parse the response
        print(response.choices[0].message.content)
        result = json.loads(response.choices[0].message.content)
        print(f"Reward model response: {result}")
        
        # Store message content for later inspection
        message_content.append({
            "model": "reward_model",
            "iteration": iteration,
            "prompt": prompt,
            "response": result
        })
        
        # Convert the result to a dictionary
        result_dict = {
            segment_data["segment_id"]: {
                "start": new_segments[segment_data["segment_id"]]["start"],
                "end": new_segments[segment_data["segment_id"]]["end"],
                "score": segment_data["score"],
                "explanation": segment_data["explanation"]
            }
            for segment_data in result["segments"]
        }

        return result_dict
    
    # Function to call the exploration agent
    def call_exploration_agent(segments_with_scores, memory_buffer):
        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Prepare the prompt
        prompt = f"""
/* Task Description */
You are a helpful assistant with access to a video that is {total_frames} frames long ({video_duration_seconds} seconds).

{question_sentence}

You are tasked with exploring the video to gather the information needed to answer a specific question with complete confidence. At each step, you may select one segment of the video to examine. Once you choose a segment, you will receive a set of representative frames sampled from that segment. Use each exploration step strategically to uncover key details, progressively refining your understanding of the video's content. Continue exploring as needed until you have acquired all the information necessary to answer the question.

In this round, you are provided with {len(segments_with_scores)} distinct segments, each covering a specific interval. The interval for each segment is detailed below.

/* Segment Information */
"""
        for segment_id, segment_data in segments_with_scores.items():
            score = segment_data["score"]
            start_time = seconds_to_timestamp(segment_data["start"])
            end_time = seconds_to_timestamp(segment_data["end"])
            prompt += f"Segment {segment_id}: [{start_time}, {end_time}] || Relevance Score: {score}%\n"
        
        prompt += f"""
/* Exploration Instruction */
For each segment, an auxiliary video assistant has already evaluated the relevance score between these frames and the question to assist you in your exploration. Focus on the segments most likely to contain key information for confidently answering the question.

Now, proceed with your exploration, selecting the segment you wish to explore or provide your final answer if you have enough information. If you want to explore more, specify which segment to explore next. If you have enough information to answer the question, provide your answer.

Hint: Do not rush to provide an answer. Take time to verify details and gather sufficient information before concluding. Approach each question as a step toward building a comprehensive understanding, ensuring accuracy over speed.

If you have enough information to answer this question, please select the best answer from the options provided.
"""
        
        # Prepare message content with memory buffer frames
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Add memory buffer frames to the message content
        for frame in memory_buffer:
            # Convert PIL image to base64
            import base64
            import io
            buffered = io.BytesIO()
            frame.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Add frame to message
            msgs[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}", "detail": "high"}
            })
        
        # Call the model
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=msgs,
            temperature=temperature,
            seed=42,
            response_format={"type": "json_schema", "json_schema": { "name": "exploration_agent", "schema": exploration_agent_schema(), "strict": True}},
            max_tokens=3000,
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        print(f"Exploration agent response: {result}")
        
        # Store message content for later inspection
        message_content.append({
            "model": "exploration_agent",
            "iteration": iteration,
            "prompt": prompt,
            "response": result
        })
        
        return result
    
    # VCA main loop
    message_content = []
    prediction_result = -1
    iteration = 0

    while True:
        print(f"Iteration {iteration + 1}")
        
        # Step 8: Uniform sampling of frames from current segment
        frames, frame_indices, new_segments, frame_paths = uniform_sample(current_segment_id, frames_per_iteration)
        
        if not frames:
            print("Warning: Could not extract any frames, skipping iteration")
            continue
        
        # Step 10: Score each segment using reward model
        segments_with_scores = call_reward_model(frames, frame_indices, new_segments, iteration)
        
        # Step 12: Update candidate segment set
        print('Segments with scores: ', segments_with_scores)
        print('New segments: ', new_segments)
        print('Candidate segments: ', candidate_segments)
        candidate_segments.update(segments_with_scores)
        
        # Step 16: Update memory with frames
        memory_buffer = update_memory(memory_buffer, frames, segments_with_scores)
        
        # Step 18: Call exploration agent to decide next action
        agent_decision = call_exploration_agent(segments_with_scores, memory_buffer)
        
        # Clean up temporary files
        for path in frame_paths:
            if os.path.exists(path):
                os.remove(path)
        
        # Step 19-24: Process agent decision
        if agent_decision["decision"] == "explore":
            # Continue exploration with selected segment
            current_segment_id = agent_decision["segment_id"]
            print(f"Selected segment {current_segment_id} for further exploration")
        else:
            # Agent has decided to answer
            answer = agent_decision["answer"]
            print(f"Agent has decided to answer: {answer}")
            
            # Convert answer to prediction result (0-4)
            option_mapping = {"Option A": 0, "Option B": 1, "Option C": 2, "Option D": 3, "Option E": 4}
            prediction_result = option_mapping.get(answer, -1)
            break
        
        iteration += 1
    
    # If we've exhausted all iterations without an answer, force a final answer
    if prediction_result == -1:
        print("Reached maximum iterations without answer, forcing final answer")
        
        # Call exploration agent one last time to get final answer
        final_decision = call_exploration_agent([], memory_buffer)
        answer = final_decision["answer"]
        
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
    Process a single video with VCA agent.
    
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

        # Execute VCA agent
        pred, message_content = execute_vca_agent()
        
        # Save results
        print(f"Results for video {video_id}: {pred}")
        save_result(os.getenv("QUESTION_FILE_PATH"), video_id, "prompt", message_content, pred, save_backup=False)
        
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
    parser = argparse.ArgumentParser(description="Run VCA agent on video datasets")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to use: egoschema, nextqa, etc.")
    parser.add_argument('--num_workers', type=int, default=None, 
                       help="Number of worker processes. Defaults to CPU count - 1")
    parser.add_argument('--partition', type=str, default="subset",
                       help="Partition to use: fullset, subset")
    args = parser.parse_args()

    # Set dataset-specific environment variables
    os.environ["DATASET"] = args.dataset
    os.environ["MODEL"] = "gpt-4o-2024-08-06"  # Lock model to gpt-4o-2024-08-06

    if args.dataset == "egoschema":
        os.environ["QUESTION_FILE_PATH"] = f"data/egoschema/{args.partition}_vca_gpt-4o-2024-08-06.json"
        os.environ["VIDEO_DIR_PATH"] = "/simurgh/u/akhatua/VideoMultiAgents/data/egoschema"
        os.environ["VIDEO_DURATIONS"] = "data/egoschema/durations.json"
    elif args.dataset == "nextqa":
        os.environ["QUESTION_FILE_PATH"] = "data/nextqa/val_vca.json"
        os.environ["VIDEO_DIR_PATH"] = "/simurgh/u/akhatua/VideoMultiAgents/data/nextqa/NExTVideo"
        os.environ["VIDEO_DURATIONS"] = "data/nextqa/durations.json"
    elif args.dataset == "momaqa":
        os.environ["QUESTION_FILE_PATH"] = "data/momaqa/test_vca.json"
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
    print(f"Using model: gpt-4o-2024-08-06")
    
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
