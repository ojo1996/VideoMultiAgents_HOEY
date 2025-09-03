import os
import time
import argparse
from multiprocessing import Pool
from functools import partial
from util import read_json_file
import traceback
from icecream import ic

from agents import VMAState, video_agent, caption_agent, graph_agent, orchestrator_agent
from langgraph.graph import StateGraph, START, END

def process_single_video(dataset, answers_json_data, video_data):
    """
    Process a single video with tools initialized inside the worker.
    
    Args:
        modality: String indicating which tools to use
        dataset: Name of the dataset being processed
        video_data: Tuple of (video_id, json_data)
    """
    video_id, json_data = video_data
    try:

        builder = StateGraph(VMAState)

        # Register nodes
        builder.add_node("video_agent", video_agent)
        builder.add_node("caption_agent", caption_agent)
        builder.add_node("graph_agent", graph_agent)
        builder.add_node("orchestrator", orchestrator_agent)

        # Fan-out from START to the three first-layer agents
        builder.add_edge(START, "video_agent")
        builder.add_edge(START, "caption_agent")
        builder.add_edge(START, "graph_agent")

        # Fan-in to orchestrator (it may run multiple times; it emits final only when ready)
        builder.add_edge("video_agent", "orchestrator")
        builder.add_edge("caption_agent", "orchestrator")
        builder.add_edge("graph_agent", "orchestrator")

        # Terminate after orchestration
        builder.add_edge("orchestrator", END)

        # Compile the system
        system = builder.compile()

        inputs = {
            "dataset": dataset,
            "video_id": video_id,
            "json_data": json_data,
            "answer_json_data": answers_json_data,

            "results_file_path": os.environ["RESULTS_FILE_PATH"]
        }

        for event in system.stream(inputs):
            print(event)
        print("Final:", system.invoke(inputs)["orchestrator_agent_answer"])
 
        return True
    except ValueError as ve:
        #print(f"Skipping video {video_id}: {ve}")
        # Gracefully continue
        return False
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        print(traceback.format_exc())
        time.sleep(1)
        return False

def get_unprocessed_videos(question_file_path, max_items=1000):
    """
    Get a list of all unprocessed videos from the question file.
    Supports both dict and list formats.
    """
    dict_data = read_json_file(question_file_path)
    unprocessed_videos = []

    # If the file is a list, enumerate over it
    for i, json_data in enumerate(dict_data[:max_items]):
        video_id = str(json_data["q_uid"])
        # Check for 'prediction' key if present
        if not ("prediction" in json_data and json_data["prediction"] != -2):
            unprocessed_videos.append((video_id, json_data))

    return unprocessed_videos

def main():
    ic.enable() 
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dataset to use for the analysis")
    parser.add_argument('--dataset', type=str, help="Example: egoschema, nextqa, etc.")
    parser.add_argument('--modality', type=str, help="Example: video, text, graph, all.")
    parser.add_argument('--agents', type=str, help="Example: single, multi_star.")
    parser.add_argument('--use_summary_info', type=bool, default=True, help="Use summary info.")
    parser.add_argument('--num_workers', type=int, default=1, 
                       help="Number of worker processes. Defaults to CPU count - 1")
    parser.add_argument('--max_items', type=int, default=999999999, 
                       help="Number of videos to process. Defaults to all.")
    parser.add_argument('--debug_sync', action='store_true', 
                       help="Run workers synchronously for debugging purposes.")
    args = parser.parse_args()

    # Set dataset-specific environment variables
    os.environ["DATASET"] = args.dataset
    if args.dataset == "egoschema":
        os.environ["QUESTION_FILE_PATH"] = f"/root/nas_Ego4D/subset.json"
        os.environ["RESULTS_FILE_PATH"] = f"/root/nas_Ego4D/results.json"
        os.environ["ANSWERS_FILE_PATH"] = f"/root/nas_Ego4D/subset_answers.json"
        os.environ["CAPTIONS_FILE"] = "/root/nas_Ego4D/egoschema_captions_gpt4o_caption_guided.json"
        os.environ["GRAPH_DATA_PATH"] = "/root/nas_Ego4D/egoschema_graph_captions.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/nas_Ego4D/egoschema_summary_cache.json"
        os.environ["VIDEO_DIR_PATH"] = "/root/nas_Ego4D/videos"
        os.environ["FRAME_NUM"] = "180"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load answers
    answers_json_data = read_json_file(os.environ["ANSWERS_FILE_PATH"])

    # Get list of unprocessed videos
    unprocessed_videos = get_unprocessed_videos(os.getenv("QUESTION_FILE_PATH"), max_items=args.max_items)
    
    # Check if debug_sync flag is set
    if args.debug_sync:
        print("Running in synchronous debug mode...")
        successful = 0
        failed = 0
        for video_data in unprocessed_videos:
            result = process_single_video(args.dataset, answers_json_data, video_data)
            if result:
                successful += 1
            else:
                failed += 1
        print(f"\nProcessing complete (synchronous mode):")
        print(f"Successfully processed: {successful} videos")
        print(f"Failed to process: {failed} videos")
    else:
        # Determine number of worker processes
        num_workers = args.num_workers
        print(f"Starting processing with {num_workers} workers")
        
        # Create process pool and process videos in parallel
        with Pool(num_workers) as pool:
            # Create a partial function with fixed arguments
            process_func = partial(process_single_video, args.dataset, answers_json_data)
            
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
