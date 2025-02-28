import os
import json
import sys
from datetime import timedelta
from langchain.agents import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import ask_gemini


@tool
def analyze_video_gemini(gpt_prompt:str) -> str:
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

    print ("gpt_prompt: ", gpt_prompt)

    video_file_name = os.getenv("VIDEO_FILE_NAME")
    video_dir       = os.getenv("VIDEO_DIR_PATH")
    if os.getenv("DATASET") == "nextqa":
        video_path = f'{video_dir}/{video_file_name.replace("-", "/")}.mp4'
    else:
        video_path = f'{video_dir}/{video_file_name}.mp4'

    print ("Called the tool of analyze_video_gemini.")

    result = ask_gemini(
                prompt_text    = gpt_prompt,
                video_path      = video_path,
                temperature    = 0.7,
            )
    print ("result: ", result)
    return result
