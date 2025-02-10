import os
import sys
import json
from langchain.agents import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import ask_gpt4_omni


def remove_bbox(data):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if key != "bbox":
                new_data[key] = remove_bbox(value)
        return new_data
    elif isinstance(data, list):
        return [remove_bbox(item) for item in data]
    else:
        return data

@tool
def analyze_video_using_graph_data(question:str) -> str:
    """
    Analyze Video using Graph Data tool.

    This tool is used to analyze the video using graph data. You can ask a question related to the Video Clip and the tool will analyze the graph data and provide the answer.

    Parameters:
    question (str): The question that you want to ask the respondent.

    Returns:
    str: Analysis result.
    """

    print("Called the Graph data tool.")

    graph_data_path  = os.getenv("GRAPH_DATA_PATH")
    graph_data_index = os.getenv("GRAPH_DATA_INDEX")

    if os.path.exists(graph_data_path) == False:
        print ("Error: The graph data file does not exist.")
        return "I am sorry, I could not find the graph data file."

    with open(graph_data_path, "r") as f:
        graph_data = json.load(f)

        graph_data_str = ""
        for item in graph_data:
            if item["file_name"] == graph_data_index:
                graph_data_str = str(remove_bbox(item))
                break

        prompt = graph_data_str
        prompt += "\n------------------------------\n"
        prompt += "Please answer the following question based on the graph data above.\n"
        prompt += f"Question : {question}\n"
        prompt += "Note : When providing answers, please generate responses based on a natural language understanding. Avoid referencing specific data structures, IDs, technical tags, or graph data, and instead use general language to explain concepts. Aim to offer explanations that can be understood without requiring background knowledge of the technical details."
        # print ("Prompt: ", prompt)

        result = ask_gpt4_omni(
            openai_api_key = os.getenv("OPENAI_API_KEY"),
            prompt_text=prompt
        )

        print ("Graph Data tool Result: ", result)

        return result


if __name__ == "__main__":

    # Set environment variables
    os.environ["GRAPH_DATA_INDEX"] = "-49z-lj8eYQ.mp4"
    os.environ["GRAPH_DATA_PATH"]  = "/root/nas_momaqa/anns/anns.json"

    data = analyze_video_using_graph_data("When the basketball player is dribbling or shooting a basket for the 1st time, What is the running match official looking at?")
    print (data)
