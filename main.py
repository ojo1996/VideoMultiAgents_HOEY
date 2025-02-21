import os
import json
import copy
import time
import random
import argparse
from util import select_data_and_mark_as_processing
from util import save_result
from util import set_environment_variables
from single_agent import execute_video_question_answering

parser = argparse.ArgumentParser(description="Dataset to use for the analysis")
parser.add_argument('--dataset', type=str, help="Example: egoschema, nextqa, etc.")
args = parser.parse_args()
dataset = args.dataset

os.environ["DATASET"] = dataset

if dataset == "egoschema":
    os.environ["QUESTION_FILE_PATH"]      = "/root/VideoMultiAgents/egoschema_fullset_anno.json"
    os.environ["CAPTIONS_FILE"]           = "/root/VideoMultiAgents/egoschema_lavila_captions.json"
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/egoschema_summary_cache.json"
    os.environ["VIDEOTREE_RESULTS_PATH"]  = "/root/VideoMultiAgents/egoschema_videotree_result.json"
    os.environ["IMAGES_DIR_PATH"]         = "/root/nas_Ego4D/egoschema/images"
    os.environ["FRAME_NUM"]               = "90"
elif dataset == "nextqa":
    os.environ["QUESTION_FILE_PATH"]      = "/root/VideoMultiAgents/nextqa_test_anno.json"
    os.environ["CAPTIONS_FILE"]           = "/root/VideoMultiAgents/nextqa_llava1.5_captions.json"
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/nextqa_summary_cache.json"
    os.environ["VIDEOTREE_RESULTS_PATH"]  = "/path/to/nextqa/frames_index"
    os.environ["IMAGES_DIR_PATH"]         = "/root/nas_nextqa/NExTVideoFrames"
    os.environ["FRAME_NUM"]               = "90"
elif dataset == "momaqa":
    os.environ["QUESTION_FILE_PATH"]      = "/root/VideoMultiAgents/momaqa_test_anno.json"
    os.environ["CAPTIONS_FILE"]           = "/root/VideoMultiAgents/momaqa_captions.json"
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/momaqa_summary_cache.json"
    os.environ["VIDEOTREE_RESULTS_PATH"]  = "/root/VideoMultiAgents/momaqa_videotree_result.json"
    os.environ["GRAPH_DATA_PATH"]         = "/root/VideoMultiAgents/momaqa_graph_data.json"
    os.environ["GRAPH_DATA_INDEX"]        = os.getenv("VIDEO_FILE_NAME")
    os.environ["IMAGES_DIR_PATH"]         = "/root/nas_momaqa/images"
    os.environ["FRAME_NUM"]               = "90"
else:
    raise ValueError(f"Unknown dataset: {dataset}")


# Sleep for a random duration (0–10 seconds) to avoid simultaneous access to the JSON file by multiple containers
sleep_time = random.uniform(0, 10)
print ("Sleeping for {} seconds".format(sleep_time))
time.sleep(sleep_time)


# Loop through questions
while True:
# for i in range(2):

    try:
        video_id, json_data = select_data_and_mark_as_processing(os.getenv("QUESTION_FILE_PATH"))
        print ("video_id: ", video_id)
        print ("json_data: ", json_data)

        if video_id is None: # All data has been processed
            break

        # Set environment variables
        print ("****************************************")
        set_environment_variables(dataset, video_id, json_data)

        result, agent_response, agent_prompts = execute_video_question_answering()

        # Save result
        print("result: ", result)
        save_result(os.getenv("QUESTION_FILE_PATH"), video_id, agent_prompts, agent_response, result, save_backup=False)

    except Exception as e:
        print ("Error: ", e)
        #unmark_as_processing(QUESTION_FILE_PATH, video_id)
        time.sleep(1)
        continue
