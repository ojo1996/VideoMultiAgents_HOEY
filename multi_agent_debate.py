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
from util import post_process, ask_gpt4_omni, create_stage2_agent_prompt, create_stage2_organizer_prompt, create_question_sentence




openai_api_key = os.getenv("OPENAI_API_KEY")


# tools = [analyze_video_gpt4o, retrieve_video_clip_captions]
tools = [analyze_video_gpt4o_with_videotree_frame_sampling, analyze_video_using_graph_data, retrieve_video_clip_captions]
# tools = [analyze_video_gpt4o, analyze_video_using_graph_data, retrieve_video_clip_captions]


#tools = [analyze_video_gpt4o, retrieve_video_clip_captions, analyze_video_based_on_the_checklist]
# tools = [analyze_video_gpt4o_with_keyword, retrieve_video_clip_captions]


llm = ChatOpenAI(
   # openai_api_key=openai_api_key,
   model="gpt-4o",
   temperature=0.0,
   # streaming=False
   )


llm_openai = ChatOpenAI(
   # openai_api_key=openai_api_key,
   model="gpt-4o",
   temperature=0.7, # o1 model only sippors temperature 1.0
   # streaming=False
   )




def create_agent(llm, tools: list, system_prompt: str):
   prompt = ChatPromptTemplate.from_messages(
       [
           SystemMessage(content="You are an expert video analyzer."),
            HumanMessage(content=system_prompt),
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
    #    print(state)
    result = agent.invoke(state)
    result["messages"].append(HumanMessage(content=result["output"], name=name))

    #    print(result)
    print ("****************************************")
    print ("result: ", result["messages"])
    return Command(
        update={
            # share internal message history with other agents
            "messages": result["messages"],
        }
    )

    # return {"messages": [HumanMessage(content=result["output"], name=name)]}




class AgentState(TypedDict):
   messages: Annotated[Sequence[BaseMessage], operator.add]
   next: str




def mas_result_to_dict(result_data):
   log_dict = {}
   for message in result_data["messages"]:
       log_dict[message.name] = message.content
   return log_dict




def execute_multi_agent_debate():
    """
    In this version, we create a 'debate_coordinator' node that decides which Agent
    will speak next (agent1, agent2, agent3), whether to bring in the organizer, or
    to finish altogether.
    """


    # --- 1) We define the members & the debate system prompt ---
    members = ["agent1", "agent2", "agent3"]
    # The debate_coordinator’s system prompt instructs it to pick who speaks next,
    # or to call the organizer, or to finish.
    system_prompt = (
        "You are the Debate Coordinator who has been tasked with answering a quiz regarding the video overseeing a debate among these members: {members}.\n"
        "They are trying to arrive at the best answer regarding the video question.\n"
        "Your job each turn is to:\n"
        " - Review everything said so far in the messages.\n"
        " - Decide who should speak next (agent1, agent2, or agent3),Each one have 1 opptunities to speak. "
        "   OR call 'organizer' if you want a final decision,\n"
        " - OR call 'FINISH' if you think the debate is done.\n"
        "Return your choice in JSON with key 'next'.\n"
    )


    # Options for 'next' in the function call
    options = members + ["organizer", "FINISH"]


    function_def = {
        "name": "route",
        "description": "Select the next role in the debate or FINISH.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}]
                }
            },
            "required": ["next"],
        },
    }


    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(
                content="Pick the next speaker from {options}, or FINISH if done.",
                additional_kwargs={"__openai_role__": "developer"}
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))


    debate_coordinator_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


    # --- 2) Build each agent (same as before, just changed prompts if needed) ---
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))


    agent1_prompt = create_stage2_agent_prompt(
        target_question_data,
        "You are Agent1, an expert in advanced video analysis. You are currently in a DEBATE"
        "with Agent2 (captions analyzer) and Agent3 (scene graph analyzer). "
    )
    agent1 = create_agent(llm_openai, [analyze_video_gpt4o], system_prompt=agent1_prompt)
    agent1_node_fn = functools.partial(agent_node, agent=agent1, name="agent1")


    agent2_prompt = create_stage2_agent_prompt(
        target_question_data,
        """
        You are Agent2, an expert in video captions and transcript interpretation. You are
        participating in a DEBATE with Agent1 (captions analyzer) and Agent3 (scene graph analyzer). 
        Step 1: Read and summarize the opinions of Agent1. Then, critically evaluate their argument.
          -  Do you agree or disagree with their reasoning? Why?
          -  Are there flaws or missing details in their logic?
          -  Provide counterarguments or refinements based on caption insights.
        Step 2: Present your own claim and justify why your perspective is stronger.
          -  Use your expertise in caption analysis to support your argument.
          -  Address any potential objections or alternative interpretations."""
    )
    agent2 = create_agent(llm_openai, [retrieve_video_clip_captions], system_prompt=agent2_prompt)
    agent2_node_fn = functools.partial(agent_node, agent=agent2, name="agent2")


    agent3_prompt = create_stage2_agent_prompt(
        target_question_data,
        """
        You are Agent3, an expert in scene-graph analysis for video. You are participating in
        a DEBATE with Agent1 (Video Analyst) and Agent2 (Captions Analyst). 
        Step 1: Review and summarize their arguments.
          -  Who do you think has a stronger argument so far?
          -  Which agent made an incorrect assumption?
          -  What is still uncertain?
        Step 2: Challenge, refine, or dispute their arguments based on scene graph analysis.
            -  Use your expertise to provide a more accurate interpretation.
            -  Address any potential objections or alternative interpretations.
        """
    )
    agent3 = create_agent(llm_openai, [retrieve_video_scene_graph], system_prompt=agent3_prompt)
    agent3_node_fn = functools.partial(agent_node, agent=agent3, name="agent3")


    # Organizer (final decider)
    organizer_prompt = create_stage2_organizer_prompt()
    organizer_agent = create_agent(llm_openai, [dummy_tool], system_prompt=organizer_prompt)
    organizer_node_fn = functools.partial(agent_node, agent=organizer_agent, name="organizer")


    agent_prompts = {
        "agent1_prompt": agent1_prompt,
        "agent2_prompt": agent2_prompt,
        "agent3_prompt": agent3_prompt,
        "organizer_prompt": organizer_prompt
    }


    print("******************** Agent1 Prompt ********************")
    print(agent1_prompt)
    print("******************** Agent2 Prompt ********************")
    print(agent2_prompt)
    print("******************** Agent3 Prompt ********************")
    print(agent3_prompt)
    print("******************** Organizer Prompt ********************")
    print(organizer_prompt)
    print("****************************************")




    # --- 3) Build the StateGraph with a debate style (not a star) ---
    workflow = StateGraph(AgentState)
    # Add the agent nodes
    workflow.add_node("agent1", agent1_node_fn)
    workflow.add_node("agent2", agent2_node_fn)
    workflow.add_node("agent3", agent3_node_fn)
    workflow.add_node("organizer", organizer_node_fn)
    # Add the debate coordinator node
    workflow.add_node("debate_coordinator", debate_coordinator_chain)


    # Whenever an Agent or the Organizer is done, go back to the coordinator
    workflow.add_edge("agent1", "debate_coordinator")
    workflow.add_edge("agent2", "debate_coordinator")
    workflow.add_edge("agent3", "debate_coordinator")
    workflow.add_edge("organizer", "debate_coordinator")


    # From the debate_coordinator, route to whichever node is chosen
    # (agent1, agent2, agent3, organizer, or FINISH)
    conditional_map = {
        "agent1": "agent1",
        "agent2": "agent2",
        "agent3": "agent3",
        "organizer": "organizer",
        "FINISH": END  # end the entire flow
    }
    workflow.add_conditional_edges(
        "debate_coordinator",
        lambda x: x["next"],
        conditional_map
    )


    # Set the entry point to the debate coordinator (so it picks who starts)
    workflow.set_entry_point("debate_coordinator")
    graph = workflow.compile()

    img = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open("graph_debate.png", "wb") as f:
        f.write(img)

    print("Graph saved to graph.png")


    # --- 4) Kick off the debate with the user’s question ---
    input_message = create_question_sentence(target_question_data)
    print("******** Debate input message **********")
    print(input_message)
    print("****************************************")


    result_data = graph.invoke(
        {
            "messages": [HumanMessage(content=input_message, name="system")],
            "next": "debate_coordinator"
        },
        {"recursion_limit": 20, "stream": False}
    )


    # Post-process result
    final_output = result_data["messages"][-1].content
    prediction_result = post_process(final_output)
    if prediction_result == -1:
        print("Result is -1. Potentially re-run or handle error.")
        return execute_multi_agent()  # or handle differently


    agents_result_dict = mas_result_to_dict(result_data)
    print("*********** Debate Structure Result **************")
    print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
    print("****************************************")


    return prediction_result, agents_result_dict, agent_prompts




if __name__ == "__main__":


   # data = {
   #     "ExpertName1": "Culinary Expert",
   #     "ExpertName1Prompt": "You are a Culinary Expert. Watch the video from the perspective of a professional chef and answer the following questions based on your expertise. Please think step-by-step.",
   #     "ExpertName2": "Kitchen Equipment Specialist",
   #     "ExpertName2Prompt": "You are a Kitchen Equipment Specialist. Watch the video from the perspective of an expert in kitchen tools and equipment and answer the following questions based on your expertise. Please think step-by-step.",
   #     "ExpertName3": "Home Cooking Enthusiast",
   #     "ExpertName3Prompt": "You are a Home Cooking Enthusiast. Watch the video from the perspective of someone who loves cooking at home and answer the following questions based on your expertise. Please think step-by-step."
   # }
  
   data = {
       "ExpertName1":"Film Studies Expert",
       "ExpertName1Prompt":"You are a Film Studies Expert. Step 1: Understand the purpose by analyzing the question about the character's primary objective in the video. Pay close attention to the narrative structure, character development, and visual storytelling elements. Evaluate the significance of each option based on how it aligns with common themes and objectives in film narratives. Step 2: Gather insights from your knowledge of film theory, narrative techniques, and character analysis, as well as any relevant insights from other experts. Step 3: Critically assess each option, considering how it fits within the context of the video’s narrative and visual cues, and rank them based on their narrative coherence. Incorporate insights from other experts to enhance your analysis. Step 4: Conclude with two detailed interpretations of the character's primary objective, explaining your reasoning with examples from the video and reflecting on any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool.",
       "ExpertName2":"Cognitive Psychologist",
       "ExpertName2Prompt":"You are a Cognitive Psychologist. Step 1: Understand the purpose by analyzing the question about the character's primary objective in the video. Focus on the character's behavior, decision-making processes, and interactions with objects. Evaluate the significance of each option based on cognitive theories of goal-directed behavior and motivation. Step 2: Gather insights from cognitive psychology, including theories of attention, perception, and motivation, as well as reflections from other experts if mentioned. Step 3: Reflect on the character's actions and choices, logically ranking each option based on its alignment with cognitive principles of goal-oriented behavior. Consider insights from other experts to provide a more comprehensive analysis. Step 4: Conclude with two detailed explanations of the character's primary objective, supported by cognitive theories and examples from the video, while incorporating any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool.",
       "ExpertName3":"Behavioral Analyst",
       "ExpertName3Prompt":"You are a Behavioral Analyst. Step 1: Understand the purpose by analyzing the question about the character's primary objective in the video. Focus on observable behaviors, patterns, and routines. Evaluate the significance of each option based on behavioral theories and the consistency of actions shown in the video. Step 2: Gather insights from behavioral analysis, including reinforcement, habit formation, and routine behaviors, as well as any relevant insights from other experts. Step 3: Reflect on the character's behavior patterns, logically ranking each option based on its consistency with behavioral principles. Incorporate insights from other experts to enhance your analysis. Step 4: Conclude with two detailed interpretations of the character's primary objective, explaining your reasoning with examples from the video and reflecting on any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool.",
   }


   execute_stage2(data)





