import os
import json
import copy
import time
import random
import argparse
from util import select_data_and_mark_as_processing
from util import create_summary_of_video
from util import save_result
from util import set_environment_variables
from stage1 import execute_stage1
from stage2 import execute_stage2

parser = argparse.ArgumentParser(description="Dataset to use for the analysis")
parser.add_argument('--dataset', type=str, help="Example: egoschema, nextqa, etc.")
args = parser.parse_args()
dataset = args.dataset

# Path to the file containing the captions data
CAPTIONS_FILE_DICT =     {
    "egoschema": "/root/nas_Ego4D/egoschema/llovi_data/egoschema/lavila_fullset.json",
    "nextqa"   : "/root/nas_nextqa/nextqa/llava1.5_fps1.json"
    }   

# Path to the file containing the questions data
QUESTION_FILE_PATH_DICT =     {
    "egoschema": "/root/nas_Ego4D/egoschema/llovi_data/egoschema/fullset_anno.json",
    "nextqa"   : "/root/nas_nextqa/nextqa/nextqa.json"
    }   

# Path to the directory containing the images  
IMAGES_DIR_PATH_DICT =     {
    "egoschema": "/root/nas_Ego4D/egoschema/images",
    "nextqa"   : "/root/nas_nextqa/NExTVideoFrames"
    }

# Number of frames to be used for the analysis
FRAME_NUM_DICT =    {
    "egoschema": 90,
    "nextqa"   : 32
    }

# Summary cache path
SUMMARY_CACHE_JSON_PATH = {
    "egoschema": "/root/VideoMultiAgents/summary_cache_egoschema.json",
    "nextqa"   : "/root/VideoMultiAgents/summary_cache_nextqa.json"
}

# VideoTree extracted frames indices path
VIDEOTREE_RESULTS_PATH_DICT =  {
    "egoschema": "/root/nas_Ego4D/egoschema/llovi_data/egoschema/VideoTree_data/depth_expansion/frames_index.json",
    "nextqa"   : "/path/to/nextqa/frames_index"
    }


os.environ["DATASET"]                 = dataset
os.environ["CAPTIONS_FILE"]           = CAPTIONS_FILE_DICT[dataset]
os.environ["QUESTION_FILE_PATH"]      = QUESTION_FILE_PATH_DICT[dataset]
os.environ["IMAGES_DIR_PATH"]         = IMAGES_DIR_PATH_DICT[dataset]
os.environ["FRAME_NUM"]               = str(FRAME_NUM_DICT[dataset])
os.environ["SUMMARY_CACHE_JSON_PATH"] = SUMMARY_CACHE_JSON_PATH[dataset]
os.environ["VIDEOTREE_RESULTS_PATH"]  = VIDEOTREE_RESULTS_PATH_DICT[dataset]

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
        # Create Summary infomation and set it as environment variable
        summary_info = create_summary_of_video(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, image_dir=os.getenv("IMAGES_DIR_PATH"), vid=os.getenv("VIDEO_INDEX"), sampling_interval_sec=1, segment_frames_num=int(os.getenv("FRAME_NUM")))
        os.environ["SUMMARY_INFO"] = json.dumps(summary_info)

        # Execute stage1
        print ("execute stage1")
        expert_info = execute_stage1()


        # Execute stage2
        print ("execute stage2")
        result, agent_response, agent_prompts = execute_stage2(expert_info)

        # Save result
        print("result: ", result)
        save_result(os.getenv("QUESTION_FILE_PATH"), video_id, expert_info, agent_prompts, agent_response, result, save_backup=False)

    except Exception as e:
        print ("Error: ", e)
        #unmark_as_processing(QUESTION_FILE_PATH, video_id)
        time.sleep(1)
        continue
