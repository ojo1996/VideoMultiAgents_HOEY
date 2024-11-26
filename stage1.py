import os
import json
import time
from util import ask_gpt4_omni
from util import create_mas_stage1_prompt
from util import extract_expert_info
from util import create_summary_of_video


def execute_stage1():
    image_dir               = os.getenv("IMAGES_DIR_PATH")
    frame_num               = int(os.getenv("FRAME_NUM"))
    openai_api_key          = os.getenv("OPENAI_API_KEY")
    video_filename          = os.getenv("VIDEO_FILE_NAME")
    qa_json_str             = os.getenv("QA_JSON_STR")

    question = json.loads(qa_json_str)
    
    # Create Summary infomation of the video
    summary_info = create_summary_of_video(openai_api_key=openai_api_key, temperature=0.7, image_dir=image_dir, vid=video_filename, sampling_interval_sec=1, segment_frames_num=frame_num)

    print ("*********** Summary Info **************")
    print(json.dumps(summary_info, indent=2, ensure_ascii=False))
    print ("****************************************")
    
    # set summary info as environment variable
    os.environ["SUMMARY_INFO"] = json.dumps(summary_info)

    prompt = create_mas_stage1_prompt(question)
    print (prompt)

    response_data = ask_gpt4_omni(
                openai_api_key = openai_api_key,
                prompt_text    = prompt,
                image_dir      = image_dir,
                vid            = video_filename,
                temperature    = 0.7,
                frame_num      = frame_num
            )
    
    print (response_data)

    expert_info = extract_expert_info(response_data)
    if not expert_info:
        print ("**** Expert info is empty. Re-running the stage1. ****")
        time.sleep(3) # sleep for 3 second to avoid the rate limit
        return execute_stage1()

    print ("*********** Stage1 Result **************")
    print(json.dumps(expert_info, indent=2, ensure_ascii=False))
    print ("****************************************")

    return expert_info


if __name__ == "__main__":

    execute_stage1()