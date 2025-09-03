import os
import json
import sys
from datetime import timedelta
from langchain.agents import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import ask_gemini


@tool
def analyze_video_gemini(analyzer_prompt:str) -> str:
    """
    Analyze video tool.

    Parameters:
    analyzer_prompt (str): Ask the analyzer questions about the video.
    The analyzer will respond each question with a step by step reasoning grounded with timestamped evidence from the video.

    Returns:
    str: The analysis result.
    """

    print ("analyzer_prompt: ", analyzer_prompt)

    video_file_name = os.getenv("VIDEO_FILE_NAME")
    video_dir       = os.getenv("VIDEO_DIR_PATH")
    if os.getenv("DATASET") == "nextqa":
        video_path = f'{video_dir}/{video_file_name.replace("-", "/")}.mp4'
    else:
        video_path = f'{video_dir}/{video_file_name}.mp4'

    print ("Called the tool of analyze_video_gemini.")

    result = ask_gemini(
                prompt_text    = analyzer_prompt,
                video_path      = video_path,
                temperature    = 0.7,
            )
    print ("result: ", result)
    return result
