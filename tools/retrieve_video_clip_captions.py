import os
import json
from datetime import timedelta
from langchain.agents import tool


@tool
def retrieve_video_clip_captions() -> list[str]:
    """
    Image captioning tool.

    Retrieve the captions of the specified video clip. Each caption is generated for notable changes within the video, helping in recognizing fine-grained changes and flow within the video. The captions include markers 'C' representing the person wearing the camera.

    Returns:
    list[str]: A list of captions for the video.
    """

    print("Called the Image captioning tool.")

    video_filename   = os.getenv("VIDEO_FILE_NAME") 
    captions_file = os.getenv("CAPTIONS_FILE")
    dataset       = os.getenv("DATASET")

    # Check if the captions file exists
    if not os.path.exists(captions_file):
        print(f"Captions file not found: {captions_file}")
        return []

    # Load the captions data
    with open(captions_file, "r") as f:
        captions_data = json.load(f)

    captions = captions_data.get(video_filename, [])
    print("Captions: ", captions)

    result = []
    previous_caption = None

    for i, caption in enumerate(captions):

        # if dataset == "egoschema":
        #     # Remove the 'C' marker from the caption
        #     caption = caption.replace("#C ", "")
        #     caption = caption.replace("#c ", "")

        # Calculate the timestamp in hh:mm:ss format
        timestamp = str(timedelta(seconds=i))

        # Add the timestamp at the beginning of each caption
        timestamped_caption = f"{timestamp}: {caption}"

        # Add the caption to the result list if it's not a duplicate of the previous one
        if caption != previous_caption:
            result.append(timestamped_caption)

        # Update the previous caption
        previous_caption = caption

    # print("Retrieved video clip captions.")
    # print(result)

    print("********************** Retrieved video clip captions.*************************")
    print(result)
    return result


if __name__ == "__main__":

    data = retrieve_video_clip_captions()
    for caption in data:
        print (caption)
    print ("length of data: ", len(data))