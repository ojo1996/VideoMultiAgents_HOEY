import os
import json
import sys
from datetime import timedelta
from langchain.agents import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import ask_gpt4_omni
from .retrieve_video_clip_captions import retrieve


@tool
def analyze_video_and_caption_gpt4o(gpt_prompt:str) -> str:
    """
    Analyze video tool.

    Parameters:
    gpt_prompt (str): In the GPT prompt, You must include 5 questions based on original questions and options.
    For example, if the question asks about the purpose of the video and OptionA is “C is looking for a T-shirt” and OptionB is “C is cleaning up the room,
    OptionA is “C is looking for a T-shirt?” and OptionB is “C is tidying the room?” and so on. 
    The questions should be Yes/No questions whenever possible.
    Also, please indicate what role you would like the respondent to play in answering the questions.

    Returns:
    str: The analysis result.
    """

    captions = '\n'.join(retrieve())

    gpt_prompt = f"\n[VIDEO CAPTIONS]\n{captions}\n\n" + gpt_prompt

    print ("gpt_prompt: ", gpt_prompt)

    openai_api_key  = os.getenv("OPENAI_API_KEY")
    video_file_name = os.getenv("VIDEO_FILE_NAME")
    frame_num       = int(os.getenv("FRAME_NUM"))
    image_dir       = os.getenv("IMAGES_DIR_PATH")

    print ("Called the tool of analyze_video_gpt4o.")

    result = ask_gpt4_omni(
                openai_api_key = openai_api_key,
                prompt_text    = gpt_prompt,
                image_dir      = image_dir,
                vid            = video_file_name,
                temperature    = 0.7,
                frame_num      = frame_num 
            )
    print ("result: ", result)
    return result
