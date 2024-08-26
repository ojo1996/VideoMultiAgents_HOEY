import os
import json
from datetime import timedelta
from langchain.agents import tool


@tool
def dummy_tool() -> str:
    """
    This is dummy tool.

    Returns:
    str: 'hello world'
    """
    print ("called the dummy tool.")
    return "hello world"


from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from util import ask_gpt4_omni

# モデルファイルのキャッシュディレクトリを設定
cache_dir = "/root/project_ws/transformers_cache"

# CLIPモデルとプロセッサを読み込む
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
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
    # ディレクトリ内の画像ファイルのリストを取得
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 画像を読み込み
    images = [Image.open(os.path.join(image_dir, file)) for file in image_files]
    
    # 画像とキーワードをプロセッサで処理
    inputs = processor(text=[keyword], images=images, return_tensors="pt", padding=True)
    
    # モデルを介したフォワードパス
    outputs = model(**inputs)
    
    # 画像とテキストの類似性スコアを取得
    logits_per_image = outputs.logits_per_image
    
    # 画像間での確率を計算
    probs_across_images = logits_per_image.softmax(dim=0)
    
    # 各画像のファイル名と確率をリストに格納
    image_probabilities = [(image_files[i], probs_across_images[i].item()) for i in range(len(image_files))]
    
    # 確率に基づいて画像をソート
    image_probabilities.sort(key=lambda x: x[1], reverse=True)
    
    # 上位N枚の画像を選択
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

    # フレームを選択
    image_dir = "/root/ms1_nas/public/Ego4D/egoschema/images"
    video_file_name = os.getenv("VIDEO_FILE_NAME")
    image_dir = os.path.join(image_dir, video_file_name)
    selected_frames = select_relevant_frames(image_dir=image_dir, keyword=keyword, top_n=90)
    
    # print(f"Selected Frames: {selected_frames}")

    # GPT-4 Omni APIにフレームを渡す
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



@tool
def analyze_video_gpt4o(gpt_prompt:str) -> str:
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

    print ("gpt_prompt: ", gpt_prompt)

    openai_api_key          = os.getenv("OPENAI_API_KEY")
    video_file_name         = os.getenv("VIDEO_FILE_NAME")

    print ("Called the tool of analyze_video_gpt4o.")

    result = ask_gpt4_omni(
                openai_api_key=openai_api_key,
                prompt_text=gpt_prompt,
                image_dir="/root/ms1_nas/public/Ego4D/egoschema/images",
                vid=video_file_name,
                temperature=0.7,
                frame_num=90
            )
    print ("result: ", result)
    return result


@tool
def retrieve_video_clip_captions(gpt_prompt:str) -> str:
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

    video_filename = os.getenv("VIDEO_FILE_NAME")

    with open("/root/project_ws/VideoMultiAgents/data/egoschema/lavila_fullset.json", "r") as f:
        captions_data = json.load(f)

    captions = captions_data.get(video_filename, [])
    result = []
    previous_caption = None

    for i, caption in enumerate(captions):

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

    openai_api_key = os.getenv("OPENAI_API_KEY")
    from util import ask_gpt4_omni
    result = ask_gpt4_omni(openai_api_key=openai_api_key, prompt_text=prompt)
    print ("result: ", result)

    return result


# previous version
@tool
def retrieve_video_clip_captions_without_llm() -> list[str]:
    """
    Image captioning tool.

    Retrieve the captions of the specified video clip. Each caption is generated for notable changes within the video, helping in recognizing fine-grained changes and flow within the video. The captions include markers 'C' representing the person wearing the camera.

    Returns:
    list[str]: A list of captions for the video.
    """

    print("Called the Image captioning tool.")

    video_filename = os.getenv("VIDEO_FILE_NAME")

    with open("/root/project_ws/VideoMultiAgents/data/egoschema/lavila_fullset.json", "r") as f:
        captions_data = json.load(f)

    captions = captions_data.get(video_filename, [])
    result = []
    previous_caption = None

    for i, caption in enumerate(captions):

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

    return result


if __name__ == "__main__":

    data = retrieve_video_clip_captions()
    for caption in data:
        print (caption)
    print ("length of data: ", len(data))