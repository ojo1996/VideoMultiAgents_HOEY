import os
import sys
import json

from util_azure import download_blob_data, save_experiment_data
# sys.path.append(os.path.abspath(".."))
from stage1 import execute_stage1
from stage2 import execute_stage2
from util import create_summary_of_video

# Check the environment variables
env_vars = [ "BLOB_CONNECTION_STRING", "COSMOS_CONNECTION_STRING", "CONTAINER_NAME", "EXPERIMENT_ID", "DATASET", "OPENAI_API_KEY", "VIDEO_FILE_NAME", "QA_JSON_STR" ]
for var in env_vars:
    value = os.getenv(var, "Not Set")
    print(f"{var}: {value}")

# Set the environment variables
os.environ["IMAGES_DIR_PATH"] = "/root/VideoMultiAgents/images"
if os.getenv("DATASET") == "egoschema":
    os.environ["CAPTIONS_FILE"] = "/root/VideoMultiAgents/egoschema_lavila_captions.json"
    os.environ["FRAME_NUM"] = str(90)
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/egoschema_summary_cache.json"
    os.environ["VIDEOTREE_RESULTS_PATH"] = "/root/VideoMultiAgents/egoschema_videotree_result.json"
    summary_info = create_summary_of_video(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, image_dir=os.getenv("IMAGES_DIR_PATH"), vid=os.getenv("VIDEO_FILE_NAME"), sampling_interval_sec=1, segment_frames_num=int(os.getenv("FRAME_NUM")))
    os.environ["SUMMARY_INFO"] = json.dumps(summary_info)
elif os.getenv("DATASET") == "nextqa":
    os.environ["CAPTIONS_FILE"] = "/root/VideoMultiAgents/nextqa_lavila_captions.json"
    os.environ["FRAME_NUM"] = str(32)
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/nextqa_summary_cache.json"
    os.environ["VIDEOTREE_RESULTS_PATH"] = "/root/VideoMultiAgents/nextqa_videotree_result.json"
    summary_info = create_summary_of_video(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, image_dir=os.getenv("IMAGES_DIR_PATH"), vid=os.getenv("VIDEO_FILE_NAME"), sampling_interval_sec=1, segment_frames_num=int(os.getenv("FRAME_NUM")))
    os.environ["SUMMARY_INFO"] = json.dumps(summary_info)
elif os.getenv("DATASET") == "momaqa":
    os.environ["CAPTIONS_FILE"] = "/root/VideoMultiAgents/momaqa_captions.json"
    os.environ["FRAME_NUM"] = str(90) # All frames
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/momaqa_summary_cache.json"
    os.environ["VIDEOTREE_RESULTS_PATH"] = "/root/VideoMultiAgents/momaqa_videotree_result.json"
    summary_info = create_summary_of_video(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, image_dir=os.getenv("IMAGES_DIR_PATH"), vid=os.getenv("VIDEO_FILE_NAME"), sampling_interval_sec=1, segment_frames_num=int(os.getenv("FRAME_NUM")))
    os.environ["SUMMARY_INFO"] = json.dumps(summary_info)

# Download images and othere necessary files
download_blob_data(os.getenv("BLOB_CONNECTION_STRING"), os.getenv("CONTAINER_NAME"), os.getenv("IMAGES_DIR_PATH"))

# Execute stage1
print ("execute stage1")
expert_info = execute_stage1()

# replace newline characters to prevent errors in json serialization
expert_info["ExpertName1Prompt"] = expert_info["ExpertName1Prompt"].replace('\n',' ')
expert_info["ExpertName2Prompt"] = expert_info["ExpertName2Prompt"].replace('\n',' ')
expert_info["ExpertName3Prompt"] = expert_info["ExpertName3Prompt"].replace('\n',' ')

# Execute stage2
print ("execute stage2")
result, agent_response, agent_prompts = execute_stage2(expert_info)
print("result: ", result)


# Save result
qa_json_data = json.loads(os.getenv("QA_JSON_STR", "{}"))
qa_json_data["id"] = os.getenv("VIDEO_FILE_NAME")
qa_json_data["video_id"] = os.getenv("CONTAINER_NAME")
qa_json_data["agent_prompts"] = agent_prompts
qa_json_data["agent_response"] = agent_response
qa_json_data["pred"] = result
save_experiment_data(os.getenv("COSMOS_CONNECTION_STRING"), os.getenv("DATASET"), os.getenv("EXPERIMENT_ID"), qa_json_data)

