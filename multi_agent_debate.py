import os
import time
import json
import operator
import functools
import sys # In some cases, you may need to import the sys module

from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod


from tools.dummy_tool import dummy_tool
from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.retrieve_video_clip_captions_with_graph_data import retrieve_video_clip_captions_with_graph_data
# from tools.retrieve_video_clip_caption_with_llm import retrieve_video_clip_caption_with_llm
from tools.analyze_video_gpt4o import analyze_video_gpt4o
# from tools.analyze_video_based_on_the_checklists import analyze_video_based_on_the_checklist
from tools.analyze_video_gpt4o_with_adaptive_frame_sampling import analyze_video_gpt4o_with_adaptive_frame_sampling
# from tools.analyze_video_gpt4o_with_keyword import analyze_video_gpt4o_with_keyword
from tools.analyze_video_using_graph_data import analyze_video_using_graph_data
from tools.analyze_video_gemini import analyze_video_gemini

from tools.analyze_video_gpt4o_with_videotree_frame_sampling import analyze_video_gpt4o_with_videotree_frame_sampling
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from util import post_process, prepare_intermediate_steps, post_intermediate_process, ask_gpt4_omni, create_organizer_prompt, create_question_sentence, create_stage2_agent_prompt




openai_api_key = os.getenv("OPENAI_API_KEY")



tools = [analyze_video_gemini, analyze_video_using_graph_data, retrieve_video_clip_captions]



llm = ChatOpenAI(
   openai_api_key=openai_api_key,
   model="gpt-4o",
   temperature=0.0,
   streaming=False
   )


llm_openai = ChatOpenAI(
   openai_api_key=openai_api_key,
   model="gpt-4o",
   temperature=0.7, # o1 model only sippors temperature 1.0
   streaming=False
   )




def create_agent(llm, tools: list, system_prompt: str, prompt: str):
   prompt = ChatPromptTemplate.from_messages(
       [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
       ]
   )
   agent = create_openai_tools_agent(llm, tools, prompt)
   executor = AgentExecutor(agent=agent, tools=tools)
   return executor



def agent1_node(state, use_summary_info):
    # print("agent1_node")
    # print("State at agent1_node start:", state)
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    if state["curr_round"] == 0:
        prompt = create_stage2_agent_prompt(
            target_question_data,
            "You are Agent1, an expert in advanced video analysis. "
            "You are currently in a DEBATE "
            "with Agent2 (captions analyzer) and Agent3 (scene graph analyzer). This is your round 1 debate. "
            "Your goal is to use your specialty to answer the question. \n"
            "You also need to decide which agent should speak next.",
            use_summary_info=use_summary_info
        )
        state["agent_prompts"]["agent1"] = prompt
    else:
        agent2 = None
        agent3 = None
        for msg in reversed(state["messages"]):
            # if msg.name == "agent1" and agent1_latest is None:
            #     agent1_latest = msg.content
            if msg.name == "agent2" and agent2 is None:
                agent2 = msg.content
            elif msg.name == "agent3" and agent3 is None:
                agent3 = msg.content

            # stop if we have them all
            if agent2 and agent3:
                break
        prompt = '''
                You are Agent1, an expert in advanced video analysis.
                You are in the second round of a DEBATE with Agent2 (Captions Analyst) and Agent3 (Scene Graph Analyst).
                You are choosen to speak in the second round of the debate.
                Could you analyze all previous agents' opinions, combing with yours own insights in the first round?
                Do you still agree with your initial claim or you change your mind
                Please provide reasons for your decision.
                Since you are already in round 2, after you speak, the next speaker will be "organizer_round2" to make a final decision.
                '''
        prompt += "\nBe sure to call the Analyze video tool. \n\n"
        prompt += "[Question]\n"
        prompt += create_question_sentence(target_question_data, False)

        state["agent_prompts"]["agent1_round2"] = prompt
    # print("agent1 prompt:", prompt)

    agent = create_agent(llm_openai, [analyze_video_gemini],
                            system_prompt="You are an expert in advanced video analysis. You are also responsieble for decide which agent should speak next.",
                            prompt=prompt)
    
    result = agent.invoke(state)
    # print("State after agent1 invocation:", result)
    name = "agent1" if state["curr_round"] == 0 else "agent1_round2"

    # Extract tool results
    intermediate_steps = prepare_intermediate_steps(result.get("intermediate_steps", []))
    output = f"Output:\n{result['output']}\n\nIntermediate Steps:\n{intermediate_steps}"

    result["messages"].append(HumanMessage(content=output, name=name))
    # print("State after agent1 invocation:", result)
    return state

def agent2_node(state, use_summary_info):
    print("agent2_node")
    # print("State at agent2_node start:", state)
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    if state["curr_round"] == 0:
        prompt = create_stage2_agent_prompt(
            target_question_data,
            "You are Agent2, an expert in video captions and transcript interpretation. This is your round 1 debate.  "
            "You are currently in a DEBATE with Agent1 (advanced video analysis) and Agent3 (scene graph analyzer).  "
            "Your goal is to use your specialty (caption interpertation) to answer the question. \n"
            "Could you also analyze and critically refute previous agnets' opinions based on your own insights from the tool?"
            "You also need to decide which agent should speak next.",
            use_summary_info=use_summary_info
        )
        state["agent_prompts"]["agent2"] = prompt
    else:
        agent1 = None
        agent3 = None
        for msg in reversed(state["messages"]):
            # if msg.name == "agent1" and agent1_latest is None:
            #     agent1_latest = msg.content
            if msg.name == "agent1" and agent1 is None:
                agent1 = msg.content
            elif msg.name == "agent3" and agent3 is None:
                agent3 = msg.content

            # stop if we have them all
            if agent1 and agent3:
                break


        prompt = f'''
            You are Agent2, an expert in video captions and transcript interpretation. You are
            participating in a DEBATE with Agent1 (captions analyzer) and Agent3 (scene graph analyzer). 
            You are choosen to speak in the second round of the debate.
            Could you analyze all previous agents' opinions, combing with yours own insights in the first round?
            Do you still agree with your initial claim or you change your mind?
            Please provide reasons for your decision.
            Since you are already in round 2, after you speak, the next speaker will be "organizer_round2" to make a final decision.
            ''' 
       
        prompt += "\nBe sure to call the Analyze video tool. \n\n"
        prompt += "[Question]\n"
        prompt += create_question_sentence(target_question_data, False)
        state["agent_prompts"]["agent2_round2"] = prompt
    # print("agent2 prompt:", prompt)
    agent = create_agent(llm_openai, [retrieve_video_clip_captions],
                            system_prompt="You are an expert in video captions and transcript interpretation.",
                            prompt=prompt)
    
    result = agent.invoke(state)
    # print("State after agent2 invocation:", result)
    name = "agent2" if state["curr_round"] == 0 else "agent2_round2"

    result["messages"].append(HumanMessage(content=result["output"], name=name))
    return state


def agent3_node(state, use_summary_info):
    print("agent3_node")
    # print("State at agent3_node start:", state)
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    if state["curr_round"] == 0:
        prompt = create_stage2_agent_prompt(
            target_question_data,
            "You are Agent3, an expert in scene-graph analysis for video. Please summarize and critique the arguments of Agent1 and Agent2. Do you agree with their conclusions? "
            "Be sure to use graph tool to analyze the question. Provide a counterargument to other agent's logic.  ",
            use_summary_info=use_summary_info
        )
        state["agent_prompts"]["agent3"] = prompt
    else:
        agent1 = None
        agent2 = None
        for msg in reversed(state["messages"]):
            if msg.name == "agent1" and agent1 is None:
                agent1 = msg.content
            elif msg.name == "agent2" and agent2 is None:
                agent2 = msg.content    
            if agent1 and agent2:
                break

        prompt = f'''
            You are Agent3, an expert in scene-graph analysis for video. 
            You are participating in a DEBATE with Agent1 and Agent2 to answer a qustion about a video. 
            You are choosen to speak in the second round of the debate.
            Could you analyze all previous agents' opinions, combing with yours own insights in the first round?
            Do you still agree with your initial claim or you change your mind? 
            Please provide reasons for your decision.
            Since you are already in round 2, after you speak, the next speaker will be "organizer_round2" to make a final decision.
            '''
            
        prompt += "\nBe sure to call the Analyze video tool. \n\n"
        prompt += "[Question]\n"
        prompt += create_question_sentence(target_question_data, False)
        state["agent_prompts"]["agent3_round2"] = prompt
    # print("agent3 prompt:", prompt)
    
    agent = create_agent(llm_openai, [retrieve_video_scene_graph],
                            system_prompt="You are an expert in scene-graph analysis for video.",
                            prompt=prompt)
    
    result = agent.invoke(state)
    # print("State after agent3 invocation:", result)
    name = "agent3" if state["curr_round"] == 0 else "agent3_round2"
    result["messages"].append(HumanMessage(content=result["output"], name=name))
    state["curr_round"] += 1
    return state

# --------------------
# 6) Aggregation Steps
# --------------------
def organizer_step(state):
    print("organizer_step")
    # print("State at organizer_step start:", state)
    """
    After each agent has spoken in a round, we gather their insights
    and produce a partial summary.
    """
    # Grab the latest outputs
    round_idx = state["curr_round"]
    agent1_latest = None
    agent2_latest = None
    agent3_latest = None

    # Walk the list in reverse to find the last message from agent1, agent2, agent3
    for msg in reversed(state["messages"]):
        if msg.name == "agent1" and agent1_latest is None:
            agent1_latest = msg.content
        elif msg.name == "agent2" and agent2_latest is None:
            agent2_latest = msg.content
        elif msg.name == "agent3" and agent3_latest is None:
            agent3_latest = msg.content

        # stop if we have them all
        if agent1_latest and agent2_latest and agent3_latest:
            break

    organizer_prompt = (
        
        f"We have completed the first round of the debate. \n\n"
        "Can you analyze all agents opinions and combine these three insights into a coherent partial summary."
        "and get an intermediate answer to the question? Also, you need to decide whether the discussion should continue or finish. "
        "If the discussion should continue, you need to decide which agent should speak next."
        "If the discussion should finish, you need to provide the final answer."
        "You are the organizer, and you have the final say in the debate. \n\n"
        "\n\n[Output Format]\n"
        "Your response should be formatted as follows:\n"
        "- Additional Discussion Needed: [YES/NO]\n"
        "- Pred: OptionX (If additional discussion is needed, provide the current leading candidate.)\n"
        "- Explanation: Provide a detailed explanation, including reasons for requiring additional discussion or the reasoning behind the final decision."
    )

    agent = create_agent(llm_openai, [dummy_tool], 
                         system_prompt="You are a skilled debate organizer combining multi-modal insights inside a debate.",
                         prompt=organizer_prompt)
    
    state["agent_prompts"]["organizer"] = organizer_prompt
    result = agent.invoke(state)
    # print("State after organizer_step invocation:", result)
    # print(f"\n[Round {round_idx + 1} Partial Summary]\n{result}\n")
    result["messages"].append(HumanMessage(content=result["output"], name="organizer"))
    # result["intermediate_result"] = post_intermediate_process(result["output"])

    # print("Intermediate result:", result["intermediate_result"])
    # sleep(100)
    return state

def organizer_final_step(state):
    print("organizer_final_step")
    print("State at organizer_final_step start:", state)
    """
    Final organizer step after all rounds are complete, merging
    all responses and producing the concluding answer.
    """
    agent1_latest = None
    agent2_latest = None
    agent3_latest = None

    # Walk the list in reverse to find the last message from agent1, agent2, agent3
    for msg in reversed(state["messages"]):
        if msg.name == "agent1" and agent1_latest is None:
            agent1_latest = msg.content
        elif msg.name == "agent2" and agent2_latest is None:
            agent2_latest = msg.content
        elif msg.name == "agent3" and agent3_latest is None:
            agent3_latest = msg.content

        # stop if we have them all
        if agent1_latest and agent2_latest and agent3_latest:
            break
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    question = create_question_sentence(target_question_data, False)
    final_prompt = create_organizer_prompt()

    agent = create_agent(llm_openai, [dummy_tool], 
                         system_prompt="You are a final organizer who produces the concluding answer. Conclude every agent's each rounds debate and opinion and provide the final answer.",
                         prompt=final_prompt)
    state["agent_prompts"]["organizer_round2"] = final_prompt
    result = agent.invoke(state)
    # print("State after organizer_final_step invocation:", result)
    result["messages"].append(HumanMessage(content=result["output"], name="organizer_round2"))
    return state

class AgentState(TypedDict):
    
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    curr_round: int
    rounds: int
    agent_prompts: Dict[str, str]
    intermediate_result: str
    # output: Dict[str, Any]
    def increment_round(self):
        self.current_round += 1

def mas_result_to_dict(result_data):
   log_dict = {}
   for message in result_data["messages"]:
       log_dict[message.name] = message.content
   return log_dict


def execute_multi_agent_multi_round(use_summary_info):
    print("execute_multi_agent_multi_round")
    members = ["agent1", "agent2", "agent3", "organizer"]
    members_round2 = [ "agent1_round2", "agent2_round2", "agent3_round2", "organizer_round2"]
    system_prompt = (
        "You are a supervisor who has been tasked with answering a quiz regarding the video in a deabte with agents. Work with the following members {members} and provide the most promising answer.\n"
        "Agent will decide who should act next and you will look at the conversation so far and choose one."
        "If all agents has spoken, you will choose the organizer to speak next."
        "Organizer will decide to continue the debate or finish the debate."
        "If the organizer decides to finish the debate, the organizer will provide the final answer and you can finish the conversation."
        "If organizer decides to continue the debate, the organizer will also decide which agent to speak next, you will then choose the agent to speak next based on organizer's response."
        "After this agent finish, you will call organizer again to make a final decision based on the current input."
        )

    options = ["FINISH"] + members + members_round2
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


    # --- 2) Build each agent (same as before, just changed prompts if needed) ---
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))


    agent1_node_fn = functools.partial(agent1_node, use_summary_info=use_summary_info)
    agent2_node_fn = functools.partial(agent2_node, use_summary_info=use_summary_info)
    agent3_node_fn = functools.partial(agent3_node, use_summary_info=use_summary_info)


    # --- 3) Build the StateGraph with a debate style (not a star) ---
    workflow = StateGraph(AgentState)
    workflow.add_node("agent1", agent1_node_fn)
    workflow.add_node("agent2", agent2_node_fn)
    workflow.add_node("agent3", agent3_node_fn)
    workflow.add_node("organizer", organizer_step)
    workflow.add_node("agent1_round2", agent1_node_fn)
    workflow.add_node("agent2_round2", agent2_node_fn)
    workflow.add_node("agent3_round2", agent3_node_fn)
    workflow.add_node("organizer_round2", organizer_final_step)
    workflow.add_node("supervisor", supervisor_chain)


    # Edges for a single round
    for member in members:
        workflow.add_edge(member, "supervisor")
    

    workflow.set_entry_point("supervisor")

  

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    # workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], {"FINISH": END, "agent1": "agent1", "agent2": "agent2", "agent3": "agent3", "organizer": "organizer", "agent1_round2": "agent1_round2", "agent2_round2": "agent2_round2", "agent3_round2": "agent3_round2", "organizer_round2": "organizer_round2"})
    workflow.add_edge("agent1_round2", "organizer_round2")
    workflow.add_edge("agent2_round2", "organizer_round2")
    workflow.add_edge("agent3_round2", "organizer_round2")
    graph = workflow.compile()

    img = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open("graph_multi_round_super.png", "wb") as f:
        f.write(img)

    print("Graph saved to graph.png")
    graph = workflow.compile()

    img = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open("graph_multi_round_new.png", "wb") as f:
        f.write(img)

    print("Graph saved to graph.png")


    input_message = create_question_sentence(target_question_data)
    
    result_data = graph.invoke(
        {
            "messages": [HumanMessage(content=input_message, name="system")],
            "next": "agent1",
            "curr_round": 0,
            "rounds": 2,
            "agent_prompts": {},
            "intermediate_result": ""
        },
        {"recursion_limit": 20, "stream": False}
    )
    # print("State after graph invocation:", result_data)


    # Post-process result
    final_output = result_data["messages"][-1].content
    prediction_result = post_process(final_output)
    organizer_step_result = ""
    for message in result_data["messages"]:
        if message.name == "organizer":
            organizer_step_result = message.content
            break
    intermediate_result = post_intermediate_process(organizer_step_result)

    print("Intermediate result::", intermediate_result)
    if prediction_result == -1:
        print("Result is -1. Potentially re-run or handle error.")
        return execute_multi_agent_multi_round()  # or handle differently


    agents_result_dict = mas_result_to_dict(result_data)
    # print("*********** Debate Structure Result **************")
    print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
    # print("****************************************")


    return prediction_result, agents_result_dict, result_data["agent_prompts"]




if __name__ == "__main__":
    print("Main execution started")


#    execute_stage2(data)
#    print("Main execution finished")
