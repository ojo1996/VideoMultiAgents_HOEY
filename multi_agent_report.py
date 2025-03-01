import os
import time
import json
import operator
import functools
from typing import Annotated, Sequence, TypedDict, List, Any
from langgraph.graph import StateGraph
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.analyze_video_gpt4o import analyze_video_gpt4o
from tools.analyze_video_gemini import analyze_video_gemini
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from tools.dummy_tool import dummy_tool
from util import post_process, ask_gpt4_omni, prepare_intermediate_steps, create_question_sentence

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-4o',
    temperature=0.0,
    disable_streaming=True
)

llm_openai = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-4o',
    temperature=0.7,
    disable_streaming=True
)

def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True) # to return intermediate steps
    return executor

def agent_node(state, agent, name):
    print("****************************************")
    print(f"Executing {name} node!")
    print (f"State: {state}")
    print("****************************************")
    result = agent.invoke(state)

    # Extract tool results
    intermediate_steps = prepare_intermediate_steps(result.get("intermediate_steps", []))

    # Combine output and intermediate steps
    combined_output = f"Output:\n{result['output']}\n\nIntermediate Steps:\n{intermediate_steps}"

    return {"messages": [HumanMessage(content=combined_output, name=name)]}

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def mas_result_to_dict(result_data):
    log_dict = {}
    for message in result_data["messages"]:
        log_dict[message.name] = message.content
    return log_dict


def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def execute_multi_agent(use_summary_info):
    # Load target question
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    video_id = os.environ["VIDEO_INDEX"]

    # Load precomputed single agent results
    base_path = "data/egoschema/"
    video_file = os.path.join(base_path, "subset_single_video.json")
    text_file = os.path.join(base_path, "subset_single_text.json")
    graph_file = os.path.join(base_path, "subset_single_graph.json")
    
    video_data = load_json_file(video_file)
    text_data = load_json_file(text_file)
    graph_data = load_json_file(graph_file)
    
    if not all([video_data, text_data, graph_data]):
        print("Error: Failed to load one or more data files.")
        return -1, {}, {}

    # Check if the video_id exists in all three datasets
    if video_id in video_data and video_id in text_data and video_id in graph_data:
        print(f'{video_id} exists in all three datasets')
        # Get predictions from each modality
        video_pred = video_data[video_id].get("pred", -1)
        text_pred = text_data[video_id].get("pred", -1)
        graph_pred = graph_data[video_id].get("pred", -1)

        print(f"video_pred: {video_pred}, text_pred: {text_pred}, graph_pred: {graph_pred}")
        
        agents_result_dict = {
            "agent1": video_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][video_pred]}") + f"\n\n{json.dumps(video_data[video_id]['response'].get('intermediate_steps', ''), indent=2)}",
            "agent2": text_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][text_pred]}"),
            "agent3": graph_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][graph_pred]}"),
            "organizer": f"All agents agree on Option {['A', 'B', 'C', 'D', 'E'][video_pred]}"
        }

        # Check if all predictions are valid
        if all(pred != -1 for pred in [video_pred, text_pred, graph_pred]):
            # Check if all agents agree
            if video_pred == text_pred == graph_pred:
                print("All agents agree! Directly returning the agreed answer.")
                prediction_result = video_pred
                
                # Create empty agent prompts dictionary
                agent_prompts = {
                    "agent1_prompt": "",
                    "agent2_prompt": "",
                    "agent3_prompt": "",
                    "organizer_prompt": ""
                }
                
                print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})")
                return prediction_result, agents_result_dict, agent_prompts


    # Use GPT-4o to analyze agent results and determine final answer
    agent_discussions = ""
    for agent in agents_result_dict:
        if agent != "organizer":
            agent_discussions += f"{agent}: {agents_result_dict[agent]}\n\n"

    gpt4o_prompt = f"""
Analyze the following multi-agent discussion and determine the final answer.
{create_question_sentence(target_question_data, shuffle_questions=False)}

Agent discussions:
{agent_discussions}

Based on the above discussion, which option is the correct answer?
Base your decision on a comprehensive analysis of each agent's opinions and the information provided.
Reason step by step to reach a decision whether the answer is one of [Option A, Option B, Option C, Option D, Option E].
"""
    
    try:
        print("******************** Organizer Prompt ********************")
        print(gpt4o_prompt)
        print("****************************************")
        gpt4o_result = ask_gpt4_omni(openai_api_key=openai_api_key, prompt_text=gpt4o_prompt, temperature=0.0)
        print("******************** Organizer Result ********************")
        print(gpt4o_result)
        print("****************************************")
        prediction_result = post_process(gpt4o_result)
    except Exception as e:
        print(f"Error using GPT-4o: {e}")
    
    # If GPT-4o failed to produce a valid result, use majority voting as fallback
    if prediction_result == -1:
        print("GPT-4o failed to produce a valid result. Using majority voting as fallback.")
        # Count votes for each option
        votes = {}
        if video_pred != -1:
            votes[video_pred] = votes.get(video_pred, 0) + 1
        if text_pred != -1:
            votes[text_pred] = votes.get(text_pred, 0) + 1
        if graph_pred != -1:
            votes[graph_pred] = votes.get(graph_pred, 0) + 1
        
        # Find the option with the most votes
        if votes:
            max_votes = max(votes.values())
            majority_options = [option for option, vote_count in votes.items() if vote_count == max_votes]
            
            # If there's a clear majority or a tie, take the first option
            prediction_result = majority_options[0]
            print(f"Majority vote result: Option {['A', 'B', 'C', 'D', 'E'][prediction_result]} with {max_votes} votes")
    
    print("****************************************")
    if os.getenv("DATASET") in ["egoschema", "nextqa"]:
        if 0 <= prediction_result <= 4:
            print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})")
        else:
            print("Error: Invalid result_data value")
    elif os.getenv("DATASET") == "momaqa":
        print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result}")
    print("****************************************")

    return prediction_result, agents_result_dict, {
        "agent1_prompt": "",
        "agent2_prompt": "",
        "agent3_prompt": "",
        "organizer_prompt": ""
    }

if __name__ == "__main__":

    execute_multi_agent()
