import os
import sys
import json

from util_azure import download_blob_data, save_experiment_data
# sys.path.append(os.path.abspath(".."))
from stage1 import execute_stage1
from stage2 import execute_stage2
from execute_video_question_answering import execute_video_question_answering
from util import get_video_summary


print("Starting the main script...")

# Check the environment variables
env_vars = [ "BLOB_CONNECTION_STRING", "COSMOS_CONNECTION_STRING", "EXPERIMENT_ID", "DATASET", "OPENAI_API_KEY", "QUESTION_ID", "VIDEO_FILE_NAME", "QA_JSON_STR" ]
for var in env_vars:
    value = os.getenv(var, "Not Set")
    print(f"{var}: {value}")


print("Environment variables checked.")

# Set the environment variables
os.environ["IMAGES_DIR_PATH"] = "/root/VideoMultiAgents/images"
if os.getenv("DATASET") == "egoschema":
    os.environ["CAPTIONS_FILE"]           = "/root/VideoMultiAgents/egoschema_graph_captions.json"
    os.environ["VIDEOTREE_RESULTS_PATH"]  = "/root/VideoMultiAgents/egoschema_videotree_result.json"
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/egoschema_summary_cache.json"
    os.environ["SUMMARY_INFO"]            = json.dumps(get_video_summary(os.getenv("SUMMARY_CACHE_JSON_PATH"), os.getenv("VIDEO_FILE_NAME")))
    os.environ["FRAME_NUM"]               = str(90)
elif os.getenv("DATASET") == "nextqa":
    os.environ["CAPTIONS_FILE"]           = "/root/VideoMultiAgents/nextqa_captions_gpt4o.json"
    os.environ["VIDEOTREE_RESULTS_PATH"]  = "/root/VideoMultiAgents/nextqa_videotree_result.json"
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/nextqa_summary_cache.json"
    os.environ["SUMMARY_INFO"]            = json.dumps(get_video_summary(os.getenv("SUMMARY_CACHE_JSON_PATH"), os.getenv("VIDEO_FILE_NAME")))
    os.environ["FRAME_NUM"]               = str(32)
elif os.getenv("DATASET") == "momaqa":
    os.environ["CAPTIONS_FILE"]           = "/root/VideoMultiAgents/momaqa_captions.json"
    os.environ["VIDEOTREE_RESULTS_PATH"]  = "/root/VideoMultiAgents/momaqa_videotree_result.json"
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/momaqa_summary_cache.json"
    os.environ["GRAPH_DATA_PATH"]         = "/root/VideoMultiAgents/momaqa_graph_data.json"
    os.environ["GRAPH_DATA_INDEX"]        = os.getenv("VIDEO_FILE_NAME")
    os.environ["SUMMARY_INFO"]            = json.dumps(get_video_summary(os.getenv("SUMMARY_CACHE_JSON_PATH"), os.getenv("VIDEO_FILE_NAME")))
    os.environ["FRAME_NUM"]               = str(90)

print("Downloaded necessary files .")

# Download images and othere necessary files
download_blob_data(os.getenv("BLOB_CONNECTION_STRING"), os.getenv("VIDEO_FILE_NAME"), os.getenv("IMAGES_DIR_PATH"))

# # Execute stage1
# print ("execute stage1")
# expert_info = execute_stage1()

# # replace newline characters to prevent errors in json serialization
# expert_info["ExpertName1Prompt"] = expert_info["ExpertName1Prompt"].replace('\n',' ')
# expert_info["ExpertName2Prompt"] = expert_info["ExpertName2Prompt"].replace('\n',' ')
# expert_info["ExpertName3Prompt"] = expert_info["ExpertName3Prompt"].replace('\n',' ')

# # Execute stage2
# print ("execute stage2")
# result, agent_response, agent_prompts = execute_stage2(expert_info)
# print("result: ", result)

# Execute Single Agent Video Question Answering
print ("Execute Single Agent Video Question Answering")
result, agent_response, agent_prompts = execute_video_question_answering()
print("result: ", result)


# Save result
qa_json_data = json.loads(os.getenv("QA_JSON_STR", "{}"))
qa_json_data["id"] = os.getenv("QUESTION_ID")
qa_json_data["video_id"] = os.getenv("VIDEO_FILE_NAME")
qa_json_data["agent_prompts"] = agent_prompts
qa_json_data["agent_response"] = agent_response
qa_json_data["pred"] = result
save_experiment_data(os.getenv("COSMOS_CONNECTION_STRING"), os.getenv("DATASET"), os.getenv("EXPERIMENT_ID"), qa_json_data)

