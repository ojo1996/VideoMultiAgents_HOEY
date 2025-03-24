import os
import json
from datetime import timedelta
from langchain.agents import tool


@tool
def retrieve_video_scene_graphs_and_enriched_captions() -> list[dict]:
    """
    Scene graph and enriched captions data tool.

    Retrieve the scene graphs information and enriched captions of the specified video clip. Each scene graph and enriched caption is generated for distinct small segments for notable changes within the video, helping in recognizing fine-grained changes and temporal flow within the video.

    Returns:
    list[dict]: A list of chunked scene graphs and enriched captions for the video.
    """

    print("Called the Scene graph and enriched captions data tool.")

    video_filename   = os.getenv("VIDEO_FILE_NAME") 
    graph_data_file = os.getenv("GRAPH_DATA_PATH")

    with open(graph_data_file,"r") as f:
        graph_data = json.load(f)
        
    chunked_graphs = graph_data[video_filename]  

    for i in chunked_graphs:
        del i["original_captions"]
        del i["yolo_detections"]
    print("********************** Retrieved video clip graph data.*************************")
    print(chunked_graphs)
    return chunked_graphs

if __name__ == "__main__":

    data = retrieve_video_scene_graphs_and_enriched_captions()
    for graph in data:
        print (graph)
    print ("length of data: ", len(data))