import os
import json
import copy
import time
import random
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from util import select_data_and_mark_as_processing
from util import save_result
from util import set_environment_variables
from util import read_json_file
from single_agent import execute_single_agent
import multi_agent_star
import multi_agent_report
import multi_agent_debate
import traceback

# Import required tools for video analysis
from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.analyze_video_gpt4o import analyze_video_gpt4o
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from tools.analyze_all_gpt4o import analyze_all_gpt4o

def get_tools(modality):
    """
    Get the appropriate tools based on the modality.
    
    Args:
        modality: String indicating which tools to use ('video', 'text', 'graph', or 'all')
    
    Returns:
        List of tool functions to use for processing
    """
    if modality == "video":
        return [analyze_video_gpt4o]
    elif modality == "text":
        return [retrieve_video_clip_captions]
    elif modality == "graph":
        return [retrieve_video_scene_graph]
    elif modality == "all":
        return [analyze_all_gpt4o]
    else:
        raise ValueError(f"Unknown modality: {modality}")

def process_single_video(modality, agents, dataset, video_data):
    """
    Process a single video with tools initialized inside the worker.
    
    Args:
        modality: String indicating which tools to use
        dataset: Name of the dataset being processed
        video_data: Tuple of (video_id, json_data)
    """
    video_id, json_data = video_data
    try:
        print(f"Processing video_id: {video_id}")
        print(f"JSON data: {json_data}")

        # Set environment variables for this process
        set_environment_variables(dataset, video_id, json_data)
        

        if agents == "single":
            # Initialize tools inside the worker process
            tools = get_tools(modality)
            # Execute video analysis
            result, agent_response, agent_prompts = execute_single_agent(tools)
        elif agents.startswith("multi-report"):
            result, agent_response, agent_prompts = multi_agent_report.execute_multi_agent()
        elif agents.startswith("multi-star"):
            result, agent_response, agent_prompts = multi_agent_star.execute_multi_agent()
        # elif agents.startswith("multi-debate"):
        #     result, agent_response, agent_prompts = multi_agent_debate.execute_multi_agent()

        # Save results
        print(f"Results for video {video_id}: {result}")
        save_result(os.getenv("QUESTION_FILE_PATH"), video_id, agent_prompts, 
                   agent_response, result, save_backup=False)
        
        return True
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        print(traceback.format_exc())
        time.sleep(1)
        return False

def get_unprocessed_videos(question_file_path, max_items=1000):
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
    parser = argparse.ArgumentParser(description="Dataset to use for the analysis")
    parser.add_argument('--dataset', type=str, help="Example: egoschema, nextqa, etc.")
    parser.add_argument('--modality', type=str, help="Example: video, text, graph, all.")
    parser.add_argument('--agents', type=str, help="Example: single, multi-star.")
    parser.add_argument('--num_workers', type=int, default=None, 
                       help="Number of worker processes. Defaults to CPU count - 1")
    args = parser.parse_args()

    # Set dataset-specific environment variables
    os.environ["DATASET"] = args.dataset
    if args.dataset == "egoschema":
        os.environ["QUESTION_FILE_PATH"] = "/root/VideoMultiAgents/egoschema_fullset_anno.json"
        os.environ["CAPTIONS_FILE"] = "/root/VideoMultiAgents/egoschema_lavila_captions.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/egoschema_summary_cache.json"
        os.environ["VIDEOTREE_RESULTS_PATH"] = "/root/VideoMultiAgents/egoschema_videotree_result.json"
        os.environ["IMAGES_DIR_PATH"] = "/root/nas_Ego4D/egoschema/images"
        os.environ["FRAME_NUM"] = "90"
    elif args.dataset == "nextqa":
        os.environ["QUESTION_FILE_PATH"] = f"data/nextqa/val_{args.agents}_{args.modality}.json"
        os.environ["GRAPH_DATA_PATH"] = "data/nextqa/nextqa_graph_captions.json"
        os.environ["CAPTIONS_FILE"] = "data/nextqa/captions_gpt4o.json"
        os.environ["IMAGES_DIR_PATH"] = "data/nextqa/frames_aligned/"
        os.environ["FRAME_NUM"] = "180"
    elif args.dataset == "momaqa":
        os.environ["QUESTION_FILE_PATH"] = "/root/VideoMultiAgents/momaqa_test_anno.json"
        os.environ["CAPTIONS_FILE"] = "/root/VideoMultiAgents/momaqa_captions.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/momaqa_summary_cache.json"
        os.environ["VIDEOTREE_RESULTS_PATH"] = "/root/VideoMultiAgents/momaqa_videotree_result.json"
        os.environ["GRAPH_DATA_PATH"] = "/root/VideoMultiAgents/momaqa_graph_data.json"
        os.environ["IMAGES_DIR_PATH"] = "/root/nas_momaqa/images"
        os.environ["FRAME_NUM"] = "90"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Get list of unprocessed videos
    unprocessed_videos = get_unprocessed_videos(os.getenv("QUESTION_FILE_PATH"))
    
    # Determine number of worker processes
    num_workers = args.num_workers

    print(f"Starting processing with {num_workers} workers")
    
    # Create process pool and process videos in parallel
    with Pool(num_workers) as pool:
        # Create a partial function with fixed arguments
        process_func = partial(process_single_video, args.modality, args.agents, args.dataset)
        
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
