import openai
import os
import json
from langchain.agents import tool



@tool
def analyze_video_gpt4o_with_videotree_frame_sampling(gpt_prompt:str) -> str:
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

    from util import ask_gpt4_omni

    video_filename = os.getenv("VIDEO_FILE_NAME") 
    question_id = os.getenv("QUESTION_ID")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    image_dir = os.getenv("IMAGES_DIR_PATH") + "/" + video_filename
    
    with open(os.getenv("VIDEOTREE_RESULTS_PATH"), 'r') as f:
        videotree_result = json.load(f)

    frame_indices = videotree_result[question_id]["sorted_values"]
    frame_indices = list(dict.fromkeys(frame_indices))
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    print("**************** VideoTree Image Paths****************")
    print(image_paths)
    selected_frames = [image_paths[idx] for idx in frame_indices]  
    print("**************** VideoTree Selected Frames****************") 
    print(selected_frames)
    
    print("**************** VideoTree Sampled Frames Indices****************")
    print(frame_indices)

    print ("Called the tool of analyze_video_gpt4o_with_videotree_frame_sampling.")

    # print("gpt_prompt: ", gpt_prompt)
    
    result = ask_gpt4_omni(
                openai_api_key=openai_api_key,
                prompt_text=gpt_prompt,
                image_dir=image_dir,
                vid=video_filename,
                temperature=0.7,
                use_selected_images=selected_frames
            )
    print ("result: ", result)
    return result