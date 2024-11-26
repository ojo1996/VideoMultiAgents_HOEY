import os
import json
import torch
import sys
from datetime import timedelta
from langchain.agents import tool
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import ask_gpt4_omni

# set the cache directory for the model files
cache_dir = "/root/project_ws/transformers_cache"

# load the CLIP model and processor
model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)


def select_relevant_frames(image_dir: str, keyword: str, top_n: int = 90) -> list:
    """
    Select the top N frames related to a specific keyword using the CLIP model.

    Parameters:
    image_dir (str): Directory containing the frames.
    keyword (str): The keyword to search for.
    top_n (int): The number of top frames to select.

    Returns:
    list: A list of file paths to the selected frames.
    """
    # get the list of image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # load the images
    images = [Image.open(os.path.join(image_dir, file)) for file in image_files]
    
    # process the images and keyword with the processor
    inputs = processor(text=[keyword], images=images, return_tensors="pt", padding=True)
    
    # forward pass through the model
    outputs = model(**inputs)
    
    # get the similarity scores between images and text
    logits_per_image = outputs.logits_per_image
    
    # calculate the probabilities across images
    probs_across_images = logits_per_image.softmax(dim=0)
    
    # create a list of tuples with image file names and probabilities
    image_probabilities = [(image_files[i], probs_across_images[i].item()) for i in range(len(image_files))]
    
    # sort the images based on probabilities
    image_probabilities.sort(key=lambda x: x[1], reverse=True)
    
    # select the top N images
    selected_files = [os.path.join(image_dir, file) for file, _ in image_probabilities[:top_n]]
    
    return selected_files


@tool
def analyze_video_gpt4o_with_keyword(gpt_prompt: str, keyword: str) -> str:
    """
    Analyze video tool with CLIP frame selection.

    Parameters:
    gpt_prompt (str): In the GPT prompt, You must include 5 questions based on original questions and options.
                    For example, if the question asks about the purpose of the video and OptionA is “C is looking for a T-shirt” and OptionB is “C is cleaning up the room,
                    OptionA is “C is looking for a T-shirt?” and OptionB is “C is tidying the room?” and so on. 
                    The questions should be Yes/No questions whenever possible.
                    Also, please indicate what role you would like the respondent to play in answering the questions.

    keyword (str): The keyword to search for relevant frames. 
                   For example, you have to use this format "a photo of a {XXXX}".
                   Note : Always include an identifiable noun, and do not use words like "action" or "C".

    Returns:
    str: The analysis result.
    """
    
    print ("Called the tool of analyze_video_gpt4o_with_keyword.")
    print ("gpt_prompt: ", gpt_prompt)
    print ("keyword: ", keyword)
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    video_file_name = os.getenv("VIDEO_FILE_NAME")

    # set the image directory
    image_dir       = os.getenv("IMAGES_DIR_PATH")
    video_file_name = os.getenv("VIDEO_FILE_NAME")
    image_dir       = os.path.join(image_dir, video_file_name)
    selected_frames = select_relevant_frames(image_dir=image_dir, keyword=keyword, top_n=90)
    
    # print(f"Selected Frames: {selected_frames}")

    # call the GPT-4o with the selected frames
    result = ask_gpt4_omni(
        openai_api_key=openai_api_key,
        prompt_text=gpt_prompt,
        image_dir=image_dir,
        vid=video_file_name,
        temperature=0.7,
        frame_num=90,
        use_selected_images=selected_frames
    )
    print("result: ", result)
    return result
