import openai
import os
import json
from langchain.agents import tool
# from .retrieve_video_clip_captions import retrieve_video_clip_captions
from .retrieve_video_clip_captions_with_gaph_data import retrieve_video_clip_captions_with_gaph_data

def adaptive_frame_sampling(image_dir: str, question:str, captions:list, video_filename:str):

    prompt = f"""
    Please determine the appropriate frame sampling method for this provided video-based question and the corresponding video's timestamped captions.\n 
    
    The captions include markers 'C' representing the person wearing the camera.\n

    question: {question}\n
    captions: {captions}\n

    Choose **only one** of the following two options and **return only the method name**:\n

    1. **Uniform Frame Sampling**: Use this method if the question requires analyzing the entire video to answer (e.g., the answer depends on information from across the video, such as a general summary or trends over time).\n
    2. **Videotree Frame Sampling**: Use this method if the question can be answered by looking at specific parts or segments of the video (e.g., the answer is based on a particular scene or events in the video).\n

    Your answer should be either: \n
    - **Videotree Frame Sampling** \n
    - **Uniform Frame Sampling**
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    attempts = 0
    retries = 3

    while attempts < retries:
            
        response = openai.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3,
        )
        
        sampling_method = response.choices[0].message.content

        if sampling_method.lower() == "videotree frame sampling":
            with open(os.getenv("VIDEOTREE_RESULTS_PATH"), 'r') as f:
                videotree_result = json.load(f)
            frame_indices = videotree_result[video_filename]["sorted_values"]
            frame_indices = list(dict.fromkeys(frame_indices))
            image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
            selected_frames = [image_paths[idx] for idx in frame_indices]  
            
            return selected_frames, frame_indices

        elif sampling_method.lower() == "uniform frame sampling":
            return None, None
            
        attempts += 1

@tool
def analyze_video_gpt4o_with_adaptive_frame_sampling(gpt_prompt:str) -> str:
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

    image_dir = os.getenv("IMAGES_DIR_PATH")
    video_filename = os.getenv("VIDEO_FILE_NAME") 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qa_json_str = os.getenv("QA_JSON_STR")
    question_data = json.loads(qa_json_str)
    question = question_data['question']

    frames = image_dir + "/" + video_filename

    # captions = retrieve_video_clip_captions({"video_index": video_filename, "captions_file": os.getenv("CAPTIONS_FILE"), "dataset": os.getenv("DATASET")})
    captions = retrieve_video_clip_captions_with_gaph_data({"video_index": video_filename, "captions_file": os.getenv("CAPTIONS_FILE"), "dataset": os.getenv("DATASET")})
    # print(captions)
    selected_frames, frame_indices = adaptive_frame_sampling(frames, question, captions, video_filename)
    frame_num = int(os.getenv("FRAME_NUM"))

    print ("Called the tool of analyze_video_gpt4o_with_adaptive_frame_sampling.")

    # if selected_frames == None:
    #     gpt_prompt += "\nThe provided frames are sampled uniformly from throughout the video and represent the information from the entire video."
    # else:
    #     indices = list([f'{frame//3600:02}:{(frame%3600)//60:02}:{frame%60:02}' for frame in frame_indices])
    #     gpt_prompt += f"\nThe provided frames are sampled from specific parts or segments of the video representing the key relevant scenes or events in the video. The selected frames are at timestamps: {indices}." 

    print("gpt_prompt: ", gpt_prompt)
    
    result = ask_gpt4_omni(
                openai_api_key=openai_api_key,
                prompt_text=gpt_prompt,
                image_dir=image_dir,
                vid=video_filename,
                temperature=0.7,
                frame_num=frame_num,
                use_selected_images=selected_frames
            )
    print ("result: ", result)
    return result