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


from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.analyze_video_gpt4o import analyze_video_gpt4o
from tools.analyze_video_gemini import analyze_video_gemini
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from tools.dummy_tool import dummy_tool
from util import post_process, create_stage2_agent_prompt, create_star_organizer_prompt, create_question_sentence, prepare_intermediate_steps

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


def execute_multi_agent(use_summary_info):
    # Load target question
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))

    # Create agents with their prompts
    agent1_prompt = create_stage2_agent_prompt(target_question_data, "You are an expert video analyzer. Follow the organizer's guidance when provided.", shuffle_questions=False, use_summary_info=use_summary_info)
    agent1 = create_agent(llm_openai, [analyze_video_gemini], system_prompt=agent1_prompt)
    agent1_node = functools.partial(agent_node, agent=agent1, name="agent1")

    agent2_prompt = create_stage2_agent_prompt(target_question_data, "You are an expert video analyzer focusing on video captions. Follow the organizer's guidance when provided.", shuffle_questions=False, use_summary_info=use_summary_info)
    agent2 = create_agent(llm_openai, [retrieve_video_clip_captions], system_prompt=agent2_prompt)
    agent2_node = functools.partial(agent_node, agent=agent2, name="agent2")

    agent3_prompt = create_stage2_agent_prompt(target_question_data, "You are an expert video analyzer focusing on video scene graphs. Follow the organizer's guidance when provided.", shuffle_questions=False, use_summary_info=use_summary_info)
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
                    "description": "Your comments on the previous agent's response and how it relates to the conversation so far. Alternatively, you can provide a final answer if you think a decision can be made based on the conversation so far."
                },
                "guidance": {
                    "title": "Guidance",
                    "type": "string",
                    "description": "Specific guidance for the next agent. Be directive and specific about what you want them to investigate or focus on."
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
                SystemMessage(
                    content="Based on the conversation so far, provide your comments on the previous agent's response and then specific guidance for the next agent. Decide which agent should speak next, or if we should provide the FINAL_ANSWER. Be directive in your guidance - tell the agent what to focus on or what specific information to look for. If final answer is decided, omit the guidance.",
                    additional_kwargs={"__openai_role__": "developer"}
                ),
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
        guidance_message = [HumanMessage(content=result["guidance"], name=f'{result["next"]}-guidance')] if result["guidance"] else []
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
    agents_result = graph.invoke(
        {"messages": [HumanMessage(content=input_message, name="system")]},
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
