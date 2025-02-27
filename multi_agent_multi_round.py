# import os
# import time
# import json
# import operator
# import functools
# import sys # In some cases, you may need to import the sys module

# from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
# from langgraph.graph import StateGraph, END
# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI

# from IPython.display import Image, display
# from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
# from langgraph.types import Command


# from tools.dummy_tool import dummy_tool
# from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
# from tools.retrieve_video_clip_captions_with_graph_data import retrieve_video_clip_captions_with_graph_data
# # from tools.retrieve_video_clip_caption_with_llm import retrieve_video_clip_caption_with_llm
# from tools.analyze_video_gpt4o import analyze_video_gpt4o
# # from tools.analyze_video_based_on_the_checklists import analyze_video_based_on_the_checklist
# from tools.analyze_video_gpt4o_with_adaptive_frame_sampling import analyze_video_gpt4o_with_adaptive_frame_sampling
# # from tools.analyze_video_gpt4o_with_keyword import analyze_video_gpt4o_with_keyword
# from tools.analyze_video_using_graph_data import analyze_video_using_graph_data
# from tools.analyze_video_gpt4o_with_videotree_frame_sampling import analyze_video_gpt4o_with_videotree_frame_sampling
# from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
# from util import post_process, post_intermediate_process, ask_gpt4_omni, create_stage2_agent_prompt, create_stage2_organizer_prompt, create_question_sentence




# openai_api_key = os.getenv("OPENAI_API_KEY")


# # tools = [analyze_video_gpt4o, retrieve_video_clip_captions]
# tools = [analyze_video_gpt4o_with_videotree_frame_sampling, analyze_video_using_graph_data, retrieve_video_clip_captions]
# # tools = [analyze_video_gpt4o, analyze_video_using_graph_data, retrieve_video_clip_captions]


# #tools = [analyze_video_gpt4o, retrieve_video_clip_captions, analyze_video_based_on_the_checklist]
# # tools = [analyze_video_gpt4o_with_keyword, retrieve_video_clip_captions]


# llm = ChatOpenAI(
#    openai_api_key=openai_api_key,
#    model="gpt-4o",
#    temperature=0.0,
#    streaming=False
#    )


# llm_openai = ChatOpenAI(
#    openai_api_key=openai_api_key,
#    model="gpt-4o",
#    temperature=0.7, # o1 model only sippors temperature 1.0
#    streaming=False
#    )




# def create_agent(llm, tools: list, system_prompt: str, prompt: str):
#    prompt = ChatPromptTemplate.from_messages(
#        [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=prompt),
#         MessagesPlaceholder(variable_name="messages"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#        ]
#    )
#    agent = create_openai_tools_agent(llm, tools, prompt)
#    executor = AgentExecutor(agent=agent, tools=tools)
#    return executor




# def agent_node(state, agent, name):
#     print ("****************************************")
#     print(f" Executing {name} node!")
#     print ("****************************************")
#     print("State before agent invocation:", state)
#     result = agent.invoke(state)
#     result["messages"].append(HumanMessage(content=result["output"], name=name))
#     print("State after agent invocation:", result)
#     print ("****************************************")
#     print ("result: ", result["messages"])
#     return Command(
#         update={
#             # share internal message history with other agents
#             "messages": result["messages"],
#         }
#     )

#     # return {"messages": [HumanMessage(content=result["output"], name=name)]}

# def agent1_node(state):
#     # print("agent1_node")
#     # print("State at agent1_node start:", state)
#     target_question_data = json.loads(os.getenv("QA_JSON_STR"))
#     if state["curr_round"] == 0:
#         prompt = create_stage2_agent_prompt(
#             target_question_data,
#             "You are Agent1, an expert in advanced video analysis. "
#             "You are currently in a DEBATE "
#             "with Agent2 (captions analyzer) and Agent3 (scene graph analyzer). This is your round 1 debate. "
#             "Your goal is to use your specialty to answer the question. \n"
#             # "Please provide your reasons and offer insightful analysis. \n"
#             # "Please think step-by-step.\n"
#         )
#         state["agent_prompts"]["agent1"] = prompt
#     else:
#         agent2 = None
#         agent3 = None
#         for msg in reversed(state["messages"]):
#             # if msg.name == "agent1" and agent1_latest is None:
#             #     agent1_latest = msg.content
#             if msg.name == "agent2" and agent2 is None:
#                 agent2 = msg.content
#             elif msg.name == "agent3" and agent3 is None:
#                 agent3 = msg.content

#             # stop if we have them all
#             if agent2 and agent3:
#                 break
#         prompt = '''
#                 You are Agent1, an expert in advanced video analysis.
#                 You are in the second round of a DEBATE with Agent2 (Captions Analyst) and Agent3 (Scene Graph Analyst).
#                 🔹 Step 1: Summarize and CRITIQUE Round 1 arguments.
#                     -    What were Agent2 and Agent3’s conclusions in round 1?
#                     -    Who made the strongest argument? Who made the weakest?
#                     -    Identify at least one contradiction in their claims.
#                 🔹 Step 2: Challenge or refine prior arguments.
#                     -    Refute or support their claims with clear reasoning.
#                     -    Provide counterexamples or highlight missing evidence in their logic.
#                 🔹 Step 3: If needed, use the Analyze Video tool.

#                     -    Before re-using tools, justify why existing evidence is insufficient.
#                     -    If you change your conclusion, explicitly state what new information made you change your stance.
#                 🚨 Do not just repeat previous findings—your goal is to engage in active debate!
#                 '''
#         # prompt = f'''
#         #     You are Agent1, an expert in advanced video analysis. You are currently in a DEBATE
#         #     with Agent2 (captions analyzer) and Agent3 (scene graph analyzer). 
#         #     Do you agree with their opinions? Please summarize the key points made by Agent2 and Agent3 in your own words.
#         #     '''
#         #  Agent 2 said {agent2}. Agent 3 said {agent3}.
#         # It's not necessary to fully agree with other's perspectives as objective is to find the correct answer.\n
#             # Please provide your reasons and offer insightful analysis to any points raised by the other Agents. 
#             # Critically assess the information they provide and either refine, refute, or support their claims with well-reasoned arguments. 
#             # Please think step-by-step.
#         # prompt += "\nBe sure to call the Analyze video tool. \n\n"
#         # prompt += "[Question]\n"
#         # prompt += create_question_sentence(target_question_data, False)
#         state["agent_prompts"]["agent1_round2"] = prompt
#     # print("agent1 prompt:", prompt)

#     agent = create_agent(llm_openai, [analyze_video_gpt4o],
#                             system_prompt="You are an expert in advanced video analysis.",
#                             prompt=prompt)
    
#     result = agent.invoke(state)
#     # print("State after agent1 invocation:", result)
#     name = "agent1_round1" if state["curr_round"] == 0 else "agent1_round2"
#     result["messages"].append(HumanMessage(content=result["output"], name=name))
#     # print("State after agent1 invocation:", result)
#     return state

# def agent2_node(state):
#     print("agent2_node")
#     # print("State at agent2_node start:", state)
#     target_question_data = json.loads(os.getenv("QA_JSON_STR"))
#     if state["curr_round"] == 0:
#         prompt = create_stage2_agent_prompt(
#             target_question_data,
#             "You are Agent2, an expert in video captions and transcript interpretation. This is your round 1 debate.  "
#             # "You are"
#             # "participating in a DEBATE with Agent1 (captions analyzer) and Agent3 (scene graph analyzer). "
#             # "Your goal is to use your specialty to answer the question about a video. \n"
#             # "Please provide your reasons and offer insightful analysis. \n"
#             # "Please think step-by-step.\n"
#         )
#         state["agent_prompts"]["agent2"] = prompt
#     else:
#         agent1 = None
#         agent3 = None
#         for msg in reversed(state["messages"]):
#             # if msg.name == "agent1" and agent1_latest is None:
#             #     agent1_latest = msg.content
#             if msg.name == "agent1" and agent1 is None:
#                 agent1 = msg.content
#             elif msg.name == "agent3" and agent3 is None:
#                 agent3 = msg.content

#             # stop if we have them all
#             if agent1 and agent3:
#                 break


#         prompt = f'''
#             You are Agent2, an expert in video captions and transcript interpretation. You are
#             participating in a DEBATE with Agent1 (captions analyzer) and Agent3 (scene graph analyzer). 
#             🔹 Step 1: Summarize and CRITIQUE Round 1 arguments.
#                 -    What were Agent1's and Agent3’s conclusions in round 1?
#                 -    Who made the strongest argument? Who made the weakest?
#                 -    Identify at least one contradiction in their claims.
#             🔹 Step 2: Defend, refine, or adjust your stance.
#                 -   Do you still agree with your initial claim?
#                 -    If another agent’s argument is stronger than yours, explain why you now lean toward their viewpoint.
#             🔹 Step 3: Consider new evidence (if necessary).
#                 -    If you call the Analyze Video tool again, justify why.
#                 -    If your conclusion changes, explicitly state why the new information overrides prior claims.
#             🚨 Your goal is not just to analyze but to directly challenge or support prior claims with stronger reasoning!
#             '''
#         # Do you agree with them? 
#         #     It's not necessary to fully agree with other's perspectives as objective is to find the correct answer.\n
#         #     Please provide your reasons and offer insightful analysis to any points raised by the other Agents. 
#         #     Critically assess the information they provide and either refine, refute, or support their claims with well-reasoned arguments. 
#         #     Please think step-by-step.
#         # Agent 1 said {agent1}. Agent 3 said {agent3}. 
#         prompt += "\nBe sure to call the Analyze video tool. \n\n"
#         prompt += "[Question]\n"
#         prompt += create_question_sentence(target_question_data, False)
#         state["agent_prompts"]["agent2_round2"] = prompt
#     # print("agent2 prompt:", prompt)
#     agent = create_agent(llm_openai, [retrieve_video_clip_captions],
#                             system_prompt="You are an expert in video captions and transcript interpretation.",
#                             prompt=prompt)
    
#     result = agent.invoke(state)
#     # print("State after agent2 invocation:", result)
#     name = "agent2_round1" if state["curr_round"] == 0 else "agent2_round2"

#     result["messages"].append(HumanMessage(content=result["output"], name=name))
#     return state


# def agent3_node(state):
#     print("agent3_node")
#     # print("State at agent3_node start:", state)
#     target_question_data = json.loads(os.getenv("QA_JSON_STR"))
#     if state["curr_round"] == 0:
#         prompt = create_stage2_agent_prompt(
#             target_question_data,
#             "You are Agent3, an expert in scene-graph analysis for video. This is your round 1 debate. "
#             # "You are participating in"
#             # "a DEBATE with Agent1 and Agent2. Your goal is to use your specialty to answer the question about a video. \n"
#             # "Please provide your reasons and offer insightful analysis. \n"
#             # "Please think step-by-step.\n"
#         )
#         state["agent_prompts"]["agent3"] = prompt
#     else:
#         agent1 = None
#         agent2 = None
#         for msg in reversed(state["messages"]):
#             if msg.name == "agent1" and agent1 is None:
#                 agent1 = msg.content
#             elif msg.name == "agent2" and agent2 is None:
#                 agent2 = msg.content    
#             if agent1 and agent2:
#                 break

#         prompt = f'''
#             You are Agent3, an expert in scene-graph analysis for video. 
#             You are participating in
#             a DEBATE with Agent1 and Agent2 to answer a qustion about a video. 
#             🔹 Step 1: Summarize and CRITIQUE Round 1 arguments.
#                 -    What were Agent1 and Agent2’s conclusions?
#                 -    Who made the strongest argument? Who made the weakest?
#                 -    Identify at least one contradiction in their claims.
#             🔹 Step 2: Provide a counterargument or refinement.
#                 -    Support or refute their logic with scene graph evidence.
#                 -    If you agree with another agent, explain why their claim is valid.
#                 -    If you disagree, explain what they misunderstood or missed.
#             🔹 Step 3: Use graph tools if necessary.
#             If you revise your answer, explain what specific insight changed your mind.
#             🚨 Your job is not just to confirm a previous answer but to DEBATE and IMPROVE the conclusion!
#             '''
#         # Do you agree with them?
#         #     First, briefly summarize the key points made by Agent1 and Agent2 use your own words.
#         #     It's not necessary to fully agree with other's perspectives as objective is to find the correct answer.\n
#         #     Please provide your reasons and offer insightful analysis to any points raised by the other Agents. 
#         #     Critically assess the information they provide and either refine, refute, or support their claims with well-reasoned arguments. 
#         #     Please think step-by-step.
#         prompt += "\nBe sure to call the Analyze video tool. \n\n"
#         prompt += "[Question]\n"
#         prompt += create_question_sentence(target_question_data, False)
#         state["agent_prompts"]["agent3_round2"] = prompt
#     # print("agent3 prompt:", prompt)
#     agent = create_agent(llm_openai, [retrieve_video_scene_graph],
#                             system_prompt="You are an expert in scene-graph analysis for video.",
#                             prompt=prompt)
    
#     result = agent.invoke(state)
#     # print("State after agent3 invocation:", result)
#     name = "agent3_round1" if state["curr_round"] == 0 else "agent3_round2"
#     result["messages"].append(HumanMessage(content=result["output"], name=name))
#     state["curr_round"] += 1
#     return state

# # --------------------
# # 6) Aggregation Steps
# # --------------------
# def aggregator_step(state):
#     print("aggregator_step")
#     # print("State at aggregator_step start:", state)
#     """
#     After each agent has spoken in a round, we gather their insights
#     and produce a partial summary.
#     """
#     # Grab the latest outputs
#     round_idx = state["curr_round"]
#     agent1_latest = None
#     agent2_latest = None
#     agent3_latest = None

#     # Walk the list in reverse to find the last message from agent1, agent2, agent3
#     for msg in reversed(state["messages"]):
#         if msg.name == "agent1" and agent1_latest is None:
#             agent1_latest = msg.content
#         elif msg.name == "agent2" and agent2_latest is None:
#             agent2_latest = msg.content
#         elif msg.name == "agent3" and agent3_latest is None:
#             agent3_latest = msg.content

#         # stop if we have them all
#         if agent1_latest and agent2_latest and agent3_latest:
#             break

#     aggregator_prompt = (
#         f"We have completed round {round_idx + 1} of the debate. \n\n"
#         "Can you analyze all agents opinions, Please combine these three insights into a coherent partial summary."
#         "and get an intermediate answer to the question?"
#         # f"The following three agents have provided analysis in round {round_idx + 1}:\n\n"
#         # f"Agent1 (Video Analysis): {agent1_latest}\n\n"
#         # f"Agent2 (Caption Interpretation): {agent2_latest}\n\n"
#         # f"Agent3 (Graph Analysis): {agent3_latest}\n\n"
#         # f"Please combine these three insights into a coherent partial summary."
#     )

#     agent = create_agent(llm_openai, [dummy_tool], 
#                          system_prompt="You are a skilled debate organizer combining multi-modal insights inside a debate.",
#                          prompt=aggregator_prompt)
    
#     result = agent.invoke(state)
#     # print("State after aggregator_step invocation:", result)
#     # print(f"\n[Round {round_idx + 1} Partial Summary]\n{result}\n")
#     result["messages"].append(HumanMessage(content=result["output"], name="aggregator"))
#     result["intermediate_result"] = result["output"]

#     # print("Intermediate result:", result["intermediate_result"])
#     return state

# def aggregator_final_step(state):
#     print("aggregator_final_step")
#     print("State at aggregator_final_step start:", state)
#     """
#     Final aggregator step after all rounds are complete, merging
#     all responses and producing the concluding answer.
#     """
#     agent1_latest = None
#     agent2_latest = None
#     agent3_latest = None

#     # Walk the list in reverse to find the last message from agent1, agent2, agent3
#     for msg in reversed(state["messages"]):
#         if msg.name == "agent1" and agent1_latest is None:
#             agent1_latest = msg.content
#         elif msg.name == "agent2" and agent2_latest is None:
#             agent2_latest = msg.content
#         elif msg.name == "agent3" and agent3_latest is None:
#             agent3_latest = msg.content

#         # stop if we have them all
#         if agent1_latest and agent2_latest and agent3_latest:
#             break
#     target_question_data = json.loads(os.getenv("QA_JSON_STR"))
#     question = create_question_sentence(target_question_data, False)
#     final_prompt = (
#         f"We have completed all 2 rounds of debate. Here are the lastest insights:\n\n"
#         f"Agent1 (Video Analysis):\n{agent1_latest}\n\n"
#         f"Agent2 (Caption Interpretation):\n{agent2_latest}\n\n"
#         f"Agent3 (Graph Analysis):\n{agent3_latest}\n\n"
#         f"Now, provide the final conclusion and answer to the question:\n"
#         f"'{question}'."
#     )

#     agent = create_agent(llm_openai, [dummy_tool], 
#                          system_prompt="You are a final aggregator who produces the concluding answer.",
#                          prompt=final_prompt)
    
#     result = agent.invoke(state)
#     # print("State after aggregator_final_step invocation:", result)
#     result["messages"].append(HumanMessage(content=result["output"], name="aggregator_final"))
#     return state

# class AgentState(TypedDict):
    
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     next: str
#     curr_round: int
#     rounds: int
#     agent_prompts: Dict[str, str]
#     intermediate_result: str
#     # output: Dict[str, Any]
#     def increment_round(self):
#         self.current_round += 1

# def mas_result_to_dict(result_data):
#    log_dict = {}
#    for message in result_data["messages"]:
#        log_dict[message.name] = message.content
#    return log_dict


# def execute_multi_agent_multi_round():
#     print("execute_multi_agent_multi_round")


#     # --- 2) Build each agent (same as before, just changed prompts if needed) ---
#     target_question_data = json.loads(os.getenv("QA_JSON_STR"))


#     agent1_node_fn = functools.partial(agent1_node)
#     agent2_node_fn = functools.partial(agent2_node)
#     agent3_node_fn = functools.partial(agent3_node)


#     # --- 3) Build the StateGraph with a debate style (not a star) ---
#     workflow = StateGraph(AgentState)
#     workflow.add_node("agent1_round1", agent1_node_fn)
#     workflow.add_node("agent2_round1", agent2_node_fn)
#     workflow.add_node("agent3_round1", agent3_node_fn)
#     workflow.add_node("aggregator_round", aggregator_step)
#     workflow.add_node("agent1_round2", agent1_node)
#     workflow.add_node("agent2_round2", agent2_node)
#     workflow.add_node("agent3_round2", agent3_node)
#     workflow.add_node("aggregator_final", aggregator_final_step)


#     # Edges for a single round
#     workflow.add_edge("agent1_round1", "agent2_round1")
#     workflow.add_edge("agent2_round1", "agent3_round1")
#     workflow.add_edge("agent3_round1", "aggregator_round")
#     workflow.add_edge("aggregator_round", "agent1_round2")
#     workflow.add_edge("agent1_round2", "agent2_round2")
#     workflow.add_edge("agent2_round2", "agent3_round2")
#     workflow.add_edge("agent3_round2", "aggregator_final")
#     workflow.add_edge("aggregator_final", END)
#     # If we still have rounds left, go to agent1, else aggregator_final
#     def aggregator_condition(state: AgentState):
#         state.increment_round()
#         if state.current_round < state.rounds:
#             return "agent1"
#         else:
#             return "aggregator_final"

#     # workflow.add_conditional_edges("aggregator_round", aggregator_condition)


#     workflow.set_entry_point("agent1_round1")
#     graph = workflow.compile()

#     # img = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
#     # with open("graph_multi_round.png", "wb") as f:
#     #     f.write(img)

#     # print("Graph saved to graph.png")


#     input_message = create_question_sentence(target_question_data)
#     # print("******** Debate input message **********")
#     # print(input_message)
#     # print("****************************************")

#     # state = AgentState([HumanMessage(content=input_message, name="system")], )
#     result_data = graph.invoke(
#         {
#             "messages": [HumanMessage(content=input_message, name="system")],
#             "next": "agent1",
#             "curr_round": 0,
#             "rounds": 2,
#             "agent_prompts": {},
#             "intermediate_result": ""
#         },
#         {"recursion_limit": 20, "stream": False}
#     )
#     # print("State after graph invocation:", result_data)


#     # Post-process result
#     final_output = result_data["messages"][-1].content
#     prediction_result = post_process(final_output)
#     intermediate_result = post_intermediate_process(result_data["intermediate_result"])

#     print("Intermediate result:", intermediate_result)
#     if prediction_result == -1:
#         print("Result is -1. Potentially re-run or handle error.")
#         return execute_multi_agent()  # or handle differently


#     agents_result_dict = mas_result_to_dict(result_data)
#     # print("*********** Debate Structure Result **************")
#     print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
#     # print("****************************************")


#     return prediction_result, agents_result_dict, result_data["agent_prompts"], intermediate_result




# if __name__ == "__main__":
#     print("Main execution started")


# #    execute_stage2(data)
# #    print("Main execution finished")



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

from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langgraph.types import Command


from tools.dummy_tool import dummy_tool
from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.retrieve_video_clip_captions_with_graph_data import retrieve_video_clip_captions_with_graph_data
# from tools.retrieve_video_clip_caption_with_llm import retrieve_video_clip_caption_with_llm
from tools.analyze_video_gpt4o import analyze_video_gpt4o
# from tools.analyze_video_based_on_the_checklists import analyze_video_based_on_the_checklist
from tools.analyze_video_gpt4o_with_adaptive_frame_sampling import analyze_video_gpt4o_with_adaptive_frame_sampling
# from tools.analyze_video_gpt4o_with_keyword import analyze_video_gpt4o_with_keyword
from tools.analyze_video_using_graph_data import analyze_video_using_graph_data
from tools.analyze_video_gpt4o_with_videotree_frame_sampling import analyze_video_gpt4o_with_videotree_frame_sampling
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from util import post_process, post_intermediate_process, ask_gpt4_omni, create_stage2_agent_prompt, create_stage2_organizer_prompt, create_question_sentence




openai_api_key = os.getenv("OPENAI_API_KEY")


# tools = [analyze_video_gpt4o, retrieve_video_clip_captions]
tools = [analyze_video_gpt4o_with_videotree_frame_sampling, analyze_video_using_graph_data, retrieve_video_clip_captions]
# tools = [analyze_video_gpt4o, analyze_video_using_graph_data, retrieve_video_clip_captions]


#tools = [analyze_video_gpt4o, retrieve_video_clip_captions, analyze_video_based_on_the_checklist]
# tools = [analyze_video_gpt4o_with_keyword, retrieve_video_clip_captions]


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



def agent1_node(state):
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
            # "Please provide your reasons and offer insightful analysis. \n"
            # "Please think step-by-step.\n"
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
                🔹 Step 1: Summarize and CRITIQUE Round 1 arguments.
                    -    What were Agent2 and Agent3’s conclusions in round 1?
                    -    Who made the strongest argument? Who made the weakest?
                    -    Identify at least one contradiction in their claims.
                🔹 Step 2: Challenge or refine prior arguments.
                    -    Refute or support their claims with clear reasoning.
                    -    Provide counterexamples or highlight missing evidence in their logic.
                🔹 Step 3: If needed, use the Analyze Video tool.

                    -    Before re-using tools, justify why existing evidence is insufficient.
                    -    If you change your conclusion, explicitly state what new information made you change your stance.
                🚨 Do not just repeat previous findings—your goal is to engage in active debate!
                '''
        # prompt = f'''
        #     You are Agent1, an expert in advanced video analysis. You are currently in a DEBATE
        #     with Agent2 (captions analyzer) and Agent3 (scene graph analyzer). 
        #     Do you agree with their opinions? Please summarize the key points made by Agent2 and Agent3 in your own words.
        #     '''
        #  Agent 2 said {agent2}. Agent 3 said {agent3}.
        # It's not necessary to fully agree with other's perspectives as objective is to find the correct answer.\n
            # Please provide your reasons and offer insightful analysis to any points raised by the other Agents. 
            # Critically assess the information they provide and either refine, refute, or support their claims with well-reasoned arguments. 
            # Please think step-by-step.
        # prompt += "\nBe sure to call the Analyze video tool. \n\n"
        # prompt += "[Question]\n"
        # prompt += create_question_sentence(target_question_data, False)
        state["agent_prompts"]["agent1_round2"] = prompt
    # print("agent1 prompt:", prompt)

    agent = create_agent(llm_openai, [analyze_video_gpt4o],
                            system_prompt="You are an expert in advanced video analysis.",
                            prompt=prompt)
    
    result = agent.invoke(state)
    # print("State after agent1 invocation:", result)
    name = "agent1_round1" if state["curr_round"] == 0 else "agent1_round2"
    result["messages"].append(HumanMessage(content=result["output"], name=name))
    # print("State after agent1 invocation:", result)
    return state

def agent2_node(state):
    print("agent2_node")
    # print("State at agent2_node start:", state)
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    if state["curr_round"] == 0:
        prompt = create_stage2_agent_prompt(
            target_question_data,
            "You are Agent2, an expert in video captions and transcript interpretation. This is your round 1 debate.  "
            # "You are"
            # "participating in a DEBATE with Agent1 (captions analyzer) and Agent3 (scene graph analyzer). "
            # "Your goal is to use your specialty to answer the question about a video. \n"
            # "Please provide your reasons and offer insightful analysis. \n"
            # "Please think step-by-step.\n"
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
            🔹 Step 1: Summarize and CRITIQUE Round 1 arguments.
                -    What were Agent1's and Agent3’s conclusions in round 1?
                -    Who made the strongest argument? Who made the weakest?
                -    Identify at least one contradiction in their claims.
            🔹 Step 2: Defend, refine, or adjust your stance.
                -   Do you still agree with your initial claim?
                -    If another agent’s argument is stronger than yours, explain why you now lean toward their viewpoint.
            🔹 Step 3: Consider new evidence (if necessary).
                -    If you call the Analyze Video tool again, justify why.
                -    If your conclusion changes, explicitly state why the new information overrides prior claims.
            🚨 Your goal is not just to analyze but to directly challenge or support prior claims with stronger reasoning!
            '''
        # Do you agree with them? 
        #     It's not necessary to fully agree with other's perspectives as objective is to find the correct answer.\n
        #     Please provide your reasons and offer insightful analysis to any points raised by the other Agents. 
        #     Critically assess the information they provide and either refine, refute, or support their claims with well-reasoned arguments. 
        #     Please think step-by-step.
        # Agent 1 said {agent1}. Agent 3 said {agent3}. 
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
    name = "agent2_round1" if state["curr_round"] == 0 else "agent2_round2"

    result["messages"].append(HumanMessage(content=result["output"], name=name))
    return state


def agent3_node(state):
    print("agent3_node")
    # print("State at agent3_node start:", state)
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    if state["curr_round"] == 0:
        prompt = create_stage2_agent_prompt(
            target_question_data,
            "You are Agent3, an expert in scene-graph analysis for video. This is your round 1 debate. "
            # "You are participating in"
            # "a DEBATE with Agent1 and Agent2. Your goal is to use your specialty to answer the question about a video. \n"
            # "Please provide your reasons and offer insightful analysis. \n"
            # "Please think step-by-step.\n"
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
            You are participating in
            a DEBATE with Agent1 and Agent2 to answer a qustion about a video. 
            🔹 Step 1: Summarize and CRITIQUE Round 1 arguments.
                -    What were Agent1 and Agent2’s conclusions?
                -    Who made the strongest argument? Who made the weakest?
                -    Identify at least one contradiction in their claims.
            🔹 Step 2: Provide a counterargument or refinement.
                -    Support or refute their logic with scene graph evidence.
                -    If you agree with another agent, explain why their claim is valid.
                -    If you disagree, explain what they misunderstood or missed.
            🔹 Step 3: Use graph tools if necessary.
            If you revise your answer, explain what specific insight changed your mind.
            🚨 Your job is not just to confirm a previous answer but to DEBATE and IMPROVE the conclusion!
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
    name = "agent3_round1" if state["curr_round"] == 0 else "agent3_round2"
    result["messages"].append(HumanMessage(content=result["output"], name=name))
    state["curr_round"] += 1
    return state

# --------------------
# 6) Aggregation Steps
# --------------------
def aggregator_step(state):
    print("aggregator_step")
    # print("State at aggregator_step start:", state)
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

    aggregator_prompt = (
        
        f"We have completed round {round_idx + 1} of the debate. \n\n"
        "Can you analyze all agents opinions, Please combine these three insights into a coherent partial summary."
        "and get an intermediate answer to the question?"
        "\n\n[Output Format]\n"
        "Your response should be formatted as follows:\n"
        "- Additional Discussion Needed: [YES/NO]\n"
        "- Pred: OptionX (If additional discussion is needed, provide the current leading candidate.)\n"
        "- Explanation: Provide a detailed explanation, including reasons for requiring additional discussion or the reasoning behind the final decision."
    )

    agent = create_agent(llm_openai, [dummy_tool], 
                         system_prompt="You are a skilled debate organizer combining multi-modal insights inside a debate.",
                         prompt=aggregator_prompt)
    
    result = agent.invoke(state)
    # print("State after aggregator_step invocation:", result)
    # print(f"\n[Round {round_idx + 1} Partial Summary]\n{result}\n")
    result["messages"].append(HumanMessage(content=result["output"], name="aggregator_round"))
    # result["intermediate_result"] = post_intermediate_process(result["output"])

    # print("Intermediate result:", result["intermediate_result"])
    # sleep(100)
    return state

def aggregator_final_step(state):
    print("aggregator_final_step")
    print("State at aggregator_final_step start:", state)
    """
    Final aggregator step after all rounds are complete, merging
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
    final_prompt = create_stage2_organizer_prompt()

    agent = create_agent(llm_openai, [dummy_tool], 
                         system_prompt="You are a final aggregator who produces the concluding answer.",
                         prompt=final_prompt)
    
    result = agent.invoke(state)
    # print("State after aggregator_final_step invocation:", result)
    result["messages"].append(HumanMessage(content=result["output"], name="aggregator_final"))
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


def execute_multi_agent_multi_round():
    print("execute_multi_agent_multi_round")
    members = ["agent1_round1", "agent2_round1", "agent3_round1", "aggregator_round", "agent1_round2", "agent2_round2", "agent3_round2", "aggregator_final"]
    system_prompt = (
        "You are a supervisor who has been tasked with answering a quiz regarding the video. Work with the following members {members} and provide the most promising answer.\n"
        "In general, there are 2 rounds of debate. Each agent will have a chance to speak in each round. "
        "After 1st round, the aggregator_round will summarize the insights and get an intermediate answer. "
        "After 2nd round, the aggregator_final will provide the final answer. "
        "In the first round, agent1, agent2, and agent3 will provide their analysis and answer independently. "
        "In the second round, they will critique each other's analysis and provide a refined answer. "
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


    # --- 2) Build each agent (same as before, just changed prompts if needed) ---
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))


    agent1_node_fn = functools.partial(agent1_node)
    agent2_node_fn = functools.partial(agent2_node)
    agent3_node_fn = functools.partial(agent3_node)


    # --- 3) Build the StateGraph with a debate style (not a star) ---
    workflow = StateGraph(AgentState)
    workflow.add_node("agent1_round1", agent1_node_fn)
    workflow.add_node("agent2_round1", agent2_node_fn)
    workflow.add_node("agent3_round1", agent3_node_fn)
    workflow.add_node("aggregator_round", aggregator_step)
    workflow.add_node("agent1_round2", agent1_node)
    workflow.add_node("agent2_round2", agent2_node)
    workflow.add_node("agent3_round2", agent3_node)
    workflow.add_node("aggregator_final", aggregator_final_step)
    workflow.add_node("supervisor", supervisor_chain)


    # Edges for a single round
    for member in members:
        workflow.add_edge(member, "supervisor")
    

    workflow.set_entry_point("supervisor")

  

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    graph = workflow.compile()

    # img = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    # with open("graph_multi_round_super.png", "wb") as f:
    #     f.write(img)

    # print("Graph saved to graph.png")


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
    aggregator_step_result = ""
    for message in result_data["messages"]:
        if message.name == "aggregator_round":
            aggregator_step_result = message.content
            break
    intermediate_result = post_intermediate_process(aggregator_step_result)

    print("Intermediate result::", intermediate_result)
    if prediction_result == -1:
        print("Result is -1. Potentially re-run or handle error.")
        return execute_multi_agent()  # or handle differently


    agents_result_dict = mas_result_to_dict(result_data)
    # print("*********** Debate Structure Result **************")
    print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
    # print("****************************************")


    return prediction_result, agents_result_dict, result_data["agent_prompts"], intermediate_result




if __name__ == "__main__":
    print("Main execution started")


#    execute_stage2(data)
#    print("Main execution finished")







