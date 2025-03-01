import os
import time
import json
import operator
import functools
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from collections import Counter


from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.analyze_video_gpt4o import analyze_video_gpt4o
from tools.analyze_video_gemini import analyze_video_gemini
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from tools.dummy_tool import dummy_tool
from util import post_process, create_agent_prompt, create_star_organizer_prompt, create_question_sentence, prepare_intermediate_steps

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
    temperature=0.7, # o1 model only sippors temperature 1.0
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
    print ("****************************************")
    print(f" Executing {name} node!")
    print ("****************************************")
    
    # Create a copy of the state to avoid modifying the original
    agent_state = state.copy()

    # Create a temporary messages list with guidance for this agent call
    agent_state["messages"] = state["messages"][-1:]
    print(f"********** {name} guidance **********")
    print(agent_state["messages"])
    print("************************************")
    
    # Invoke the agent with the temporary state
    result = agent.invoke(agent_state)

    # # Extract tool results
    # intermediate_steps = prepare_intermediate_steps(result.get("intermediate_steps", []))

    # # Combine output and intermediate steps
    # combined_output = f"Output:\n{result['output']}\n\nIntermediate Steps:\n{intermediate_steps}"

    # return {"messages": [HumanMessage(content=combined_output, name=name)]}

    return {"messages": [HumanMessage(content=result["output"], name=name)]}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def mas_result_to_dict(result_data):
    log_dict = {}
    
    for message in result_data["messages"]:
        base_name = message.name
        # Create a unique name if needed
        if base_name in log_dict:
            index = 2
            new_name = f"{base_name}-{index}"
            while new_name in log_dict:
                index += 1
                new_name = f"{base_name}-{index}"
            log_dict[new_name] = message.content
        else:
            log_dict[base_name] = message.content
    
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
        
        # Check if all predictions are valid
        if all(pred != -1 for pred in [video_pred, text_pred, graph_pred]):
            # Check if all agents agree
            if video_pred == text_pred == graph_pred:
                print("All agents agree! Directly returning the agreed answer.")
                prediction_result = video_pred
                
                # Create a simplified result dictionary
                agents_result_dict = {
                    "agent1": video_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][video_pred]}") + f"\n\n{json.dumps(video_data[video_id]['response'].get('intermediate_steps', ''), indent=2)}",
                    "agent2": text_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][text_pred]}"),
                    "agent3": graph_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][graph_pred]}"),
                    "organizer": f"All agents agree on Option {['A', 'B', 'C', 'D', 'E'][video_pred]}"
                }
                
                # Create empty agent prompts dictionary
                agent_prompts = {
                    "agent1_prompt": "",
                    "agent2_prompt": "",
                    "agent3_prompt": "",
                    "organizer_prompt": ""
                }
                
                print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})")
                return prediction_result, agents_result_dict, agent_prompts

    # If we reach here, either the video_id doesn't exist in all datasets,
    # or the agents don't agree, so we proceed with the Star process
    
    # Create agents with their prompts
    agent1_prompt = create_agent_prompt(target_question_data, agent_type="video_expert", use_summary_info=use_summary_info)
    agent1 = create_agent(llm_openai, [analyze_video_gemini], system_prompt=agent1_prompt)
    agent1_node = functools.partial(agent_node, agent=agent1, name="agent1")

    agent2_prompt = create_agent_prompt(target_question_data, agent_type="text_expert", use_summary_info=use_summary_info)
    agent2 = create_agent(llm_openai, [retrieve_video_clip_captions], system_prompt=agent2_prompt)
    agent2_node = functools.partial(agent_node, agent=agent2, name="agent2")

    agent3_prompt = create_agent_prompt(target_question_data, agent_type="graph_expert", use_summary_info=use_summary_info)
    agent3 = create_agent(llm_openai, [retrieve_video_scene_graph], system_prompt=agent3_prompt)
    agent3_node = functools.partial(agent_node, agent=agent3, name="agent3")

    # Create organizer with a central role
    organizer_prompt = create_star_organizer_prompt()
    
    # Organizer options now include END to directly finish the process
    organizer_options = ["agent1", "agent2", "agent3", "FINAL_ANSWER"]
    organizer_function_def = {
        "name": "route",
        "description": "Select the next agent to speak or provide final answer.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {"title": "Next", "anyOf": [{"enum": organizer_options}]},
                "comment": {
                    "title": "Comment", 
                    "type": "string",
                    "description": "Your comments on the previous agent's response and how it relates to the conversation so far. Alternatively, you can provide a final answer if you think a decision can be made based on the conversation so far. Your final answer should be one of the following options: OptionA, OptionB, OptionC, OptionD, OptionE, along with an explanation."
                },
                "guidance": {
                    "title": "Guidance",
                    "type": "string",
                    "description": "Specific guidance for the next agent, if you choose to ask another agent. Be directive about what information is needed or what aspects to investigate. Focus on requesting objective analysis rather than suggesting specific conclusions. Ask for information or analysis without implying expected outcomes."
                }
            },
            "required": ["next", "comment", "guidance"],
        },
    }
    
    # Define organizer node that will decide which agent speaks next
    def organizer_node(state):
        print ("****************************************")
        print(" Executing organizer node!")
        print ("****************************************")
        
        # Process the conversation so far
        organizer_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=organizer_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(options=str(organizer_options))

        # Print the rendered prompt template for debugging
        rendered_prompt = organizer_prompt_template.format_messages(messages=state["messages"])
        print("************* Rendered Organizer Prompt **************")
        for message in rendered_prompt:
            print(f"Role: {message.type}")
            print(f"Content: {message.content}")
            print("---")
        print("****************************************")
        
        organizer_chain = (
            organizer_prompt_template
            | llm_openai.bind_functions(functions=[organizer_function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )
        
        result = organizer_chain.invoke(state)

        print ("************* Organizer Result **************")
        print (result)
        print ("****************************************")
        
        # Add organizer's comments to the conversation
        guidance_message = [HumanMessage(content=result["guidance"], name=f'{result["next"]}-guidance')] if result["next"] != 'FINAL_ANSWER' else []
        return {
            "messages": [HumanMessage(content=result["comment"], name="organizer")] + guidance_message,
            "next": result["next"]
        }

    # for debugging
    agent_prompts = {
        "agent1_prompt": agent1_prompt,
        "agent2_prompt": agent2_prompt,
        "agent3_prompt": agent3_prompt,
        "organizer_prompt": organizer_prompt
    }

    print ("******************** Agent1 Prompt ********************")
    print (agent1_prompt)
    print ("******************** Agent2 Prompt ********************")
    print (agent2_prompt)
    print ("******************** Agent3 Prompt ********************")
    print (agent3_prompt)
    print ("******************** Organizer Prompt ********************")
    print (organizer_prompt)
    print ("****************************************")
    # return

    # Create the workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent1", agent1_node)
    workflow.add_node("agent2", agent2_node)
    workflow.add_node("agent3", agent3_node)
    workflow.add_node("organizer", organizer_node)

    # Add edges to the workflow - organizer is central
    workflow.add_edge("agent1", "organizer")
    workflow.add_edge("agent2", "organizer")
    workflow.add_edge("agent3", "organizer")
    
    # Organizer decides which agent speaks next or when to finish
    workflow.add_conditional_edges(
        "organizer",
        lambda x: x["next"],
        {"agent1": "agent1", "agent2": "agent2", "agent3": "agent3", "FINAL_ANSWER": END}
    )
    
    # Set entry point to organizer
    workflow.set_entry_point("organizer")
    graph = workflow.compile()

    # Execute the graph
    input_message = create_question_sentence(target_question_data)
    print ("******** Multiagent input_message **********")
    print (input_message)
    print ("****************************************")
    
    # Initialize with the question and set next to organizer
    # If we have precomputed results, add them to the initial messages
    initial_messages = [HumanMessage(content=input_message, name="system")]
    
    if video_id in video_data and "response" in video_data[video_id] and "output" in video_data[video_id]["response"]:
        initial_messages.append(HumanMessage(content='Output: ' + video_data[video_id]["response"]["output"] + '\n\nIntermediate Steps: ' + json.dumps(video_data[video_id]["response"]["intermediate_steps"]), name="agent1"))
    
    if video_id in text_data and "response" in text_data[video_id] and "output" in text_data[video_id]["response"]:
        initial_messages.append(HumanMessage(content=text_data[video_id]["response"]["output"], name="agent2"))
    
    if video_id in graph_data and "response" in graph_data[video_id] and "output" in graph_data[video_id]["response"]:
        initial_messages.append(HumanMessage(content=graph_data[video_id]["response"]["output"], name="agent3"))
    
    agents_result = graph.invoke(
        {"messages": initial_messages},
        {"recursion_limit": 20, "stream": False}
    )

    prediction_result = post_process(agents_result["messages"][-1].content)
    if prediction_result == -1:
        print ("***********************************************************")
        print ("Error: The result is -1. So, retry the stage2.")
        print ("***********************************************************")
        time.sleep(1)
        return execute_multi_agent(use_summary_info)

    agents_result_dict = mas_result_to_dict(agents_result)

    print ("*********** Multiagent Result **************")
    print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
    print ("****************************************")
    if os.getenv("DATASET") == "egoschema" or os.getenv("DATASET") == "nextqa":
        print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result} (Option{['A', 'B', 'C', 'D', 'E'][prediction_result]})" if 0 <= prediction_result <= 4 else "Error: Invalid result_data value")
    elif os.getenv("DATASET") == "momaqa":
        print (f"Truth: {target_question_data['truth']}, Pred: {prediction_result}")
    print ("****************************************")

    return prediction_result, agents_result_dict, agent_prompts


if __name__ == "__main__":

    execute_multi_agent(use_summary_info=True)
