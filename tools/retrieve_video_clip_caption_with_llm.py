import os
import json
import sys
from datetime import timedelta
from langchain.agents import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import ask_gpt4_omni


@tool
def retrieve_video_clip_caption_with_llm(gpt_prompt:str) -> str:
    """
    Analyze captioning tool.

    Parameters:
    gpt_prompt (str): In the GPT prompt, You must include 5 questions based on original questions and options.
    For example, if the question asks about the purpose of the video and OptionA is “C is looking for a T-shirt” and OptionB is “C is cleaning up the room,
    OptionA is “C is looking for a T-shirt?” and OptionB is “C is tidying the room?” and so on. 
    The questions should be Yes/No questions whenever possible.
    Also, please indicate what role you would like the respondent to play in answering the questions.

    Returns:
    str: The analysis result.
    """

    print("Called the Image captioning tool.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    video_index    = os.getenv("VIDEO_INDEX")
    captions_file  = os.getenv("CAPTIONS_FILE")
    dataset        = os.getenv("DATASET")

    with open(captions_file, "r") as f:
        captions_data = json.load(f)

    captions = captions_data.get(video_index, [])
    result = []
    previous_caption = None

    for i, caption in enumerate(captions):

        if dataset == "egoschema":
            # Remove the 'C' marker from the caption
            caption = caption.replace("#C ", "")
            caption = caption.replace("#c ", "")

        # Calculate the timestamp in hh:mm:ss format
        timestamp = str(timedelta(seconds=i))

        # Add the timestamp at the beginning of each caption
        timestamped_caption = f"{timestamp}: {caption}"

        # Add the caption to the result list if it's not a duplicate of the previous one
        if caption != previous_caption:
            result.append(timestamped_caption)

        # Update the previous caption
        previous_caption = caption

    prompt = "[Image Captions]\n"
    for caption in result:
        prompt += caption + "\n"

    prompt += "\n[Instructions]\n"
    prompt += gpt_prompt    

    # print ("gpt_prompt: ", prompt)

    result = ask_gpt4_omni(openai_api_key=openai_api_key, prompt_text=prompt)
    print ("result: ", result)

    return result
