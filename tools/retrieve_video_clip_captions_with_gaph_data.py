import os
import json
from datetime import timedelta
from langchain.agents import tool


@tool
def retrieve_video_clip_captions_with_gaph_data() -> list[str]:
    """
    Image captioning tool.

    Retrieve the captions of the specified video clip. Each caption is generated for distinct small segments for notable changes within the video, helping in recognizing fine-grained changes and flow within the video. The captions include markers 'C' representing the person wearing the camera.

    Returns:
    list[str]: A list of captions for the video.
    """

    print("Called the Image captioning tool from inside the tool")

    video_index   = os.getenv("VIDEO_INDEX")
    print("video_index: ", video_index)
    captions_file = os.getenv("CAPTIONS_FILE")
    print("captions_file: ", captions_file)
    dataset       = os.getenv("DATASET")
    print("dataset: ", dataset)
    with open(captions_file,"r") as f:
        captions_data = json.load(f)
        
    caption = captions_data[video_index]  

    timestamped_captions = []

    for i in caption:
        start = i["time_start"]
        end = i["time_end"]
        enriched_caption = i["enriched_caption"]
        timestamp = f'{start//3600:02}:{(start%3600)//60:02}:{start%60:02}-{end//3600:02}:{(end%3600)//60:02}:{end%60:02}'
        timestamped_caption = f"{timestamp}: {enriched_caption}"
        timestamped_captions.append(timestamped_caption)

    print(timestamped_captions)
    return timestamped_captions


if __name__ == "__main__":

    data = retrieve_video_clip_captions_with_gaph_data()
    for caption in data:
        print (caption)
    print ("length of data: ", len(data))