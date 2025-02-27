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
from util import post_process, create_stage2_agent_prompt, create_stage2_organizer_prompt, create_question_sentence, prepare_intermediate_steps

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
    result = agent.invoke(state)

    # Extract tool results
    intermediate_steps = prepare_intermediate_steps(result.get("intermediate_steps", []))

    # Combine output and intermediate steps
    combined_output = f"Output:\n{result['output']}\n\nIntermediate Steps:\n{intermediate_steps}"

    return {"messages": [HumanMessage(content=combined_output, name=name)]}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def mas_result_to_dict(result_data):
    log_dict = {}
    for message in result_data["messages"]:
        log_dict[message.name] = message.content
    return log_dict


def execute_multi_agent(use_summary_info):
    members = ["agent1", "agent2", "agent3", "organizer"]
    system_prompt = (
        "You are a supervisor who has been tasked with answering a quiz regarding the video. Work with the following members {members} and provide the most promising answer.\n"
        "Respond with FINISH along with your final answer. Each agent has one opportunity to speak, and the organizer should make the final decision."
        )

    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(
                content="Given the conversation above, who should act next? Or should we FINISH? Select one of: {options} If you want to finish the conversation, type 'FINISH' and Final Answer.",
                additional_kwargs={"__openai_role__": "developer"}
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    # Load taget question
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))

    # print ("****************************************")
    # print (" Next Question: {}".format(os.getenv("VIDEO_FILE_NAME")))
    # print ("****************************************")
    # print (create_question_sentence(target_question_data))

    agent1_prompt = create_stage2_agent_prompt(target_question_data, "You are an expert video analyzer.", shuffle_questions=False, use_summary_info=use_summary_info)
    agent1 = create_agent(llm_openai, [analyze_video_gemini], system_prompt=agent1_prompt)
    agent1_node = functools.partial(agent_node, agent=agent1, name="agent1")

    agent2_prompt = create_stage2_agent_prompt(target_question_data, "You are an expert video analyzer.", shuffle_questions=False, use_summary_info=use_summary_info)
    agent2 = create_agent(llm_openai, [retrieve_video_clip_captions], system_prompt=agent2_prompt)
    agent2_node = functools.partial(agent_node, agent=agent2, name="agent2")

    agent3_prompt = create_stage2_agent_prompt(target_question_data, "You are an expert video analyzer.", shuffle_questions=False, use_summary_info=use_summary_info)
    agent3 = create_agent(llm_openai, [retrieve_video_scene_graph], system_prompt=agent3_prompt)
    agent3_node = functools.partial(agent_node, agent=agent3, name="agent3")

    organizer_prompt = create_stage2_organizer_prompt()
    organizer_agent = create_agent(llm_openai, [dummy_tool], system_prompt=organizer_prompt)
    organizer_node = functools.partial(agent_node, agent=organizer_agent, name="organizer")

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
    workflow.add_node("supervisor", supervisor_chain)

    # Add edges to the workflow
    for member in members:
        workflow.add_edge(member, "supervisor")
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("supervisor")
    graph = workflow.compile()

    # Execute the graph
    input_message = create_question_sentence(target_question_data)
    print ("******** Multiagent input_message **********")
    print (input_message)
    print ("****************************************")
    agents_result = graph.invoke(
        {"messages": [HumanMessage(content=input_message, name="system")], "next": "agent1"},
        {"recursion_limit": 20, "stream": False}
    )

    prediction_result = post_process(agents_result["messages"][-1].content)
    if prediction_result == -1:
        print ("***********************************************************")
        print ("Error: The result is -1. So, retry the stage2.")
        print ("***********************************************************")
        time.sleep(1)
        return execute_multi_agent()

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

    execute_multi_agent()
