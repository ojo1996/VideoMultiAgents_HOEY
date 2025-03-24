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
    if os.getenv("DATASET") == "nextqa":
        video_id = target_question_data["q_uid"]
    elif os.getenv("DATASET") == "egoschema":
        video_id = os.getenv("VIDEO_INDEX")

    # Load precomputed single agent results
    base_path = "data/results/"
    if os.getenv("DATASET") == "nextqa":
        video_file = os.path.join(base_path, "nextqa_val_single_video.json")
        text_file = os.path.join(base_path, "nextqa_val_single_text.json")
        graph_file = os.path.join(base_path, "nextqa_val_single_graph.json")
    elif os.getenv("DATASET") == "egoschema":
        video_file = os.path.join(base_path, "egoschema_fullset_single_video.json")
        text_file = os.path.join(base_path, "egoschema_fullset_single_text.json")
        graph_file = os.path.join(base_path, "egoschema_fullset_single_graph.json")
    
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
    organizer_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "organizer",
            "schema": {"type": "object",
                       "properties": {
                           "reasoning": {"type": "string"},
                           "answer": {"type": "string", "enum": ["Option A", "Option B", "Option C", "Option D", "Option E"]}
                        },
                        "required": ["reasoning", "answer"],
                        "additionalProperties": False
                       },
            "strict": True
        }
    }
    
    try:
        print("******************** Organizer Prompt ********************")
        print(gpt4o_prompt)
        print("****************************************")
        gpt4o_result = ask_gpt4_omni(openai_api_key=openai_api_key, prompt_text=gpt4o_prompt, temperature=0.0, json_schema=organizer_schema)
        print("******************** Organizer Result ********************")
        print(gpt4o_result)
        print("****************************************")
        agents_result_dict["organizer"] = gpt4o_result
        prediction_result = post_process(json.loads(gpt4o_result)["answer"])
    except Exception as e:
        print(f"Error using GPT-4o: {e}")
    
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
