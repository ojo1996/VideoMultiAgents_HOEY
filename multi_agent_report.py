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
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from tools.dummy_tool import dummy_tool
from util import post_process, create_agent_prompt, create_organizer_prompt, create_question_sentence, prepare_intermediate_steps

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

def execute_multi_agent():
    # Load the question data from an environment variable
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))

    # Create prompts for each agent
    agent1_prompt    = create_agent_prompt(target_question_data, agent_type="video_expert")
    agent2_prompt    = create_agent_prompt(target_question_data, agent_type="text_expert")
    agent3_prompt    = create_agent_prompt(target_question_data, agent_type="graph_expert")
    organizer_prompt = create_organizer_prompt()

    # Create agents
    agent1 = create_agent(llm_openai, [analyze_video_gpt4o], system_prompt=agent1_prompt)
    agent1_node = functools.partial(agent_node, agent=agent1, name="agent1")

    agent2 = create_agent(llm_openai, [retrieve_video_clip_captions], system_prompt=agent2_prompt)
    agent2_node = functools.partial(agent_node, agent=agent2, name="agent2")

    agent3 = create_agent(llm_openai, [retrieve_video_scene_graph], system_prompt=agent3_prompt)
    agent3_node = functools.partial(agent_node, agent=agent3, name="agent3")

    organizer_agent = create_agent(llm_openai, [dummy_tool], system_prompt=organizer_prompt)
    organizer_node = functools.partial(agent_node, agent=organizer_agent, name="organizer")

    # Print prompts
    print("******************** Agent1 Prompt ********************")
    print(agent1_prompt)
    print("******************** Agent2 Prompt ********************")
    print(agent2_prompt)
    print("******************** Agent3 Prompt ********************")
    print(agent3_prompt)
    print("******************** Organizer Prompt ********************")
    print(organizer_prompt)
    print("****************************************")

    # agent1 → agent2 → agent3 → organizer
    workflow = StateGraph(AgentState)
    workflow.add_node("agent1", agent1_node)
    workflow.add_node("agent2", agent2_node)
    workflow.add_node("agent3", agent3_node)
    workflow.add_node("organizer", organizer_node)

    workflow.add_edge("agent1", "agent2")
    workflow.add_edge("agent2", "agent3")
    workflow.add_edge("agent3", "organizer")

    workflow.set_entry_point("agent1")
    graph = workflow.compile()

    # prepare input data and invoke
    input_message = create_question_sentence(target_question_data)
    print("******** Multiagent input_message **********")
    print(input_message)
    print("****************************************")
    agents_result = graph.invoke({"messages": [HumanMessage(content=input_message, name="system")]}, {"recursion_limit": 20, "stream": False})

    print (agents_result)

    prediction_result = post_process(agents_result["messages"][-1].content)
    if prediction_result == -1:
        print("***********************************************************")
        print("Error: The result is -1. Retrying stage2.")
        print("***********************************************************")
        time.sleep(1)
        return execute_multi_agent()

    agents_result_dict = mas_result_to_dict(agents_result)

    print("*********** Multiagent Result **************")
    print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
    print("****************************************")
    if os.getenv("DATASET") in ["egoschema", "nextqa"]:
        if 0 <= prediction_result <= 4:
            print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result} (Option{['A', 'B', 'C', 'D', 'E'][prediction_result]})")
        else:
            print("Error: Invalid result_data value")
    elif os.getenv("DATASET") == "momaqa":
        print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result}")
    print("****************************************")

    return prediction_result, agents_result_dict, {
        "agent1_prompt": agent1_prompt,
        "agent2_prompt": agent2_prompt,
        "agent3_prompt": agent3_prompt,
        "organizer_prompt": organizer_prompt
    }

if __name__ == "__main__":

    execute_multi_agent()
