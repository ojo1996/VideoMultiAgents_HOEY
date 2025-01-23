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
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# for llama3
# import transformers
# import torch
# from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain_huggingface import ChatHuggingFace

from tools.dummy_tool import dummy_tool
from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
# from tools.retrieve_video_clip_caption_with_llm import retrieve_video_clip_caption_with_llm
from tools.analyze_video_gpt4o import analyze_video_gpt4o
# from tools.analyze_video_based_on_the_checklists import analyze_video_based_on_the_checklist
# from tools.analyze_video_gpt4o_with_adaptive_frame_sampling import analyze_video_gpt4o_with_adaptive_frame_sampling
# from tools.analyze_video_gpt4o_with_keyword import analyze_video_gpt4o_with_keyword

from util import post_process, ask_gpt4_omni, create_stage2_agent_prompt, create_stage2_organizer_prompt, create_question_sentence


openai_api_key = os.getenv("OPENAI_API_KEY")

tools = [analyze_video_gpt4o, retrieve_video_clip_captions]
#tools = [analyze_video_gpt4o, retrieve_video_clip_captions, analyze_video_based_on_the_checklist]
# tools = [analyze_video_gpt4o_with_keyword, retrieve_video_clip_captions]

llm = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-4o',
    temperature=0.0,
    streaming=False
    )

llm_openai = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-4o',
    temperature=0.7,
    streaming=False
    )

# groq_api_key = os.getenv("GROQ_API_KEY")
# llm_groq = ChatGroq(
#     temperature=0,
#     model="llama-3.1-70b-versatile",
#     # model="llama3-groq-70b-8192-tool-use-preview",
#     # model="llama3-8b-8192",
#     api_key=groq_api_key
# )

# llm_ollama = ChatOllama(
#     model="llama3.1:70b",
#     temperature=0.0,
#     streaming=False
#     )


def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ( "system", system_prompt, ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    print ("****************************************")
    print(f" Executing {name} node!")
    print ("****************************************")
    result = agent.invoke(state)
    # print ("****************************************")
    # print ("result: ", result["output"])
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def mas_result_to_dict(result_data):
    log_dict = {}
    for message in result_data["messages"]:
        log_dict[message.name] = message.content
    return log_dict


def execute_stage2(expert_info):

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
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}"
                " If you want to finish the conversation, type 'FINISH' and Final Answer."
                ,
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

    agent1_prompt = create_stage2_agent_prompt(target_question_data, expert_info["ExpertName1Prompt"], shuffle_questions=False)
    agent1 = create_agent(llm_openai, tools, system_prompt=agent1_prompt)
    agent1_node = functools.partial(agent_node, agent=agent1, name="agent1")

    agent2_prompt = create_stage2_agent_prompt(target_question_data, expert_info["ExpertName2Prompt"], shuffle_questions=False)
    agent2 = create_agent(llm_openai, tools, system_prompt=agent2_prompt)
    agent2_node = functools.partial(agent_node, agent=agent2, name="agent2")

    agent3_prompt = create_stage2_agent_prompt(target_question_data, expert_info["ExpertName3Prompt"], shuffle_questions=False)
    agent3 = create_agent(llm_openai, [retrieve_video_clip_captions], system_prompt=agent3_prompt)
    agent3_node = functools.partial(agent_node, agent=agent3, name="agent3")

    organizer_prompt = create_stage2_organizer_prompt(target_question_data, shuffle_questions=False)
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
    print ("******** Stage2 input_message **********")
    print (input_message)
    print ("****************************************")
    agents_result = graph.invoke(
        {"messages": [HumanMessage(content=input_message, name="system")], "next": "agent1"},
        {"recursion_limit": 20}
    )

    prediction_num = post_process(agents_result["messages"][-1].content)
    if prediction_num == -1:
        prompt = agents_result["messages"][-1].content + "\n\nPlease retrieve the final answer from the sentence above. Your response should be one of the following options: Option A, Option B, Option C, Option D, Option E."
        response_data = ask_gpt4_omni(openai_api_key=openai_api_key, prompt_text=prompt)
        prediction_num = post_process(response_data)
    if prediction_num == -1:
        print ("***********************************************************")
        print ("Error: The result is -1. So, retry the stage2.")
        print ("***********************************************************")
        time.sleep(1)
        return execute_stage2(expert_info)

    agents_result_dict = mas_result_to_dict(agents_result)

    print ("*********** Stage2 Result **************")
    print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
    print ("****************************************")
    print(f"Truth: {target_question_data['truth']}, Pred: {prediction_num} (Option{['A', 'B', 'C', 'D', 'E'][prediction_num]})" if 0 <= prediction_num <= 4 else "Error: Invalid result_data value")
    print ("****************************************")

    return prediction_num, agents_result_dict, agent_prompts


if __name__ == "__main__":

    data = {
        "ExpertName1": "Culinary Expert",
        "ExpertName1Prompt": "You are a Culinary Expert. Watch the video from the perspective of a professional chef and answer the following questions based on your expertise. Please think step-by-step.",
        "ExpertName2": "Kitchen Equipment Specialist",
        "ExpertName2Prompt": "You are a Kitchen Equipment Specialist. Watch the video from the perspective of an expert in kitchen tools and equipment and answer the following questions based on your expertise. Please think step-by-step.",
        "ExpertName3": "Home Cooking Enthusiast",
        "ExpertName3Prompt": "You are a Home Cooking Enthusiast. Watch the video from the perspective of someone who loves cooking at home and answer the following questions based on your expertise. Please think step-by-step."
    }

    execute_stage2(data)