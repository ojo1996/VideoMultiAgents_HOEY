import os
import json
import sys
from datetime import timedelta
from langchain.agents import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@tool
def retrieve_video_scene_graph() -> str:
    """
    Scene graph tool.

    Retrieve the scene graphs information of the specified video clip. Each scene graph is generated for distinct small segments for notable changes within the video, helping in recognizing fine-grained changes and temporal flow within the video.

    Returns:
    list[dict]: A list of chunked scene graphs for the video.
    """
    return retrieve_graph()


def retrieve_graph() -> str:
    video_filename   = os.getenv("VIDEO_FILE_NAME") 
    graph_data_file = os.getenv("GRAPH_DATA_PATH")
    dataset       = os.getenv("DATASET")

    with open(graph_data_file,"r") as f:
        captions_data = json.load(f)

    if video_filename not in captions_data:
        print(f"Video {video_filename} not found in the scene graph data")
        return []

    caption = captions_data[video_filename]

    timestamped_graph = []

    for i in caption:
        start = i["time_start"]
        end = i["time_end"]
        graph = i["scene_graph"]
        graph_text = '\n'.join(' '.join(t) for t in graph)
        timestamp = f'{start//3600:02}:{(start%3600)//60:02}:{start%60:02}-{end//3600:02}:{(end%3600)//60:02}:{end%60:02}'
        timestamped_graph.append(f"{timestamp}: {graph_text}")
    
    timestamped_graph = '\n'.join(timestamped_graph)

    print("********************** Retrieved video scene graph *************************")
    print(timestamped_graph)
    
    return timestamped_graph
