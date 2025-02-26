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
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
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

from util import post_process, ask_gpt4_omni, create_stage2_agent_prompt, create_question_sentence, create_report_organizer_prompt


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


def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            # HumanMessage(content=system_prompt), 
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    print("PROMPt::")
    print(prompt)
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    print ("****************************************")
    print(f" Executing {name} node!")
    print ("****************************************111")
    print(state["messages"] )
    result = agent.invoke(state)
    print ("****************************************")
    # print ("result: ", result)
    result["messages"].append(HumanMessage(content=result["output"], name=name))
    print ("result MESSAGES: ", result["messages"])
    return Command(
        update={
            # share internal message history with other agents
            "messages": result["messages"],
        }
    )

  

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def mas_result_to_dict(result_data):
    log_dict = {}
    for message in result_data["messages"]:
        log_dict[message.name] = message.content
    return log_dict


def execute_multi_agent_report():
    
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    agent1_prompt = create_stage2_agent_prompt(target_question_data, 
    '''
    You are agent1. You are an expert in advanced video analysis. Your primary task is to analyze the visual aspects of the video, including objects, actions, events, and their spatial-temporal relationships. "
    Your findings will serve as the foundation for further analysis, so ensure your descriptions are clear and detailed. 
    Focus on key visual elements that could be relevant to answering the question.
    ''',
    shuffle_questions=False)

    agent1 = create_agent(llm_openai, [analyze_video_gpt4o], system_prompt=agent1_prompt)
    agent1_node = functools.partial(agent_node, agent=agent1, name="agent1")

    agent2_prompt = create_stage2_agent_prompt(target_question_data, 
    '''
    You are agent2. You are an expert in video captions and transcript interpretation. Your role is to integrate textual information from captions and transcripts. "
    Then, use the given captions to ensure that your textual analysis aligns with the visual details provided. \n
    \n\n### Instructions:\n
    1. **Summarize Agent1's findings** in your own words to confirm your understanding.\n
    2. **Cross-check the captions and transcript** with the visual analysis based on the questions:\n
       - Identify matching details.\n
       - Spot inconsistencies or additional insights not mentioned by Agent1.\n
    3. **Think step by step** to systematically integrate textual and visual details based on the questions.\n
    ''' ,
    shuffle_questions=False)
    agent2 = create_agent(llm_openai, [retrieve_video_clip_captions], system_prompt=agent2_prompt)
    agent2_node = functools.partial(agent_node, agent=agent2, name="agent2")

    agent3_prompt = create_stage2_agent_prompt(target_question_data, 
    '''
    You are Agent3, an expert in scene-graph analysis for video. You must fitst carefully analyze Agent2's findings and refine the overall interpretation of the scene.
    \n\n### Instructions:\n
    1. Summarize the findings of **Agent2 (caption interpretation)** in your own words to confirm your understanding.\n
    2. Construct a **scene graph representation** based on the questions, focusing on:\n
       - Entities (e.g., objects, people, locations)\n
       - Relationships between them (e.g., holding, walking towards, standing next to)\n
       - Any temporal or spatial context relevant to the question\n
    3. **Think step by step** to systematically analyze your answer to the question.
    ''',
    shuffle_questions=False)
    agent3 = create_agent(llm_openai, [retrieve_video_scene_graph], system_prompt=agent3_prompt)
    agent3_node = functools.partial(agent_node, agent=agent3, name="agent3")
   
    organizer_prompt = create_report_organizer_prompt()
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
    # workflow.add_node("supervisor", supervisor_chain)

    # Add edges to the workflow
    # for member in members:
    #     workflow.add_edge(member, "supervisor")
    workflow.add_edge("agent1", "agent2")
    workflow.add_edge("agent2", "agent3")
    workflow.add_edge("agent3", "organizer")
    # Once the organizer finishes, we end the workflow
    workflow.add_edge("organizer", END)

    workflow.set_entry_point("agent1")
    graph = workflow.compile()

    
    img = graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API)
    with open("graph_report.png", "wb") as f:
        f.write(img)

    print("Graph saved to graph.png")
  
    # Execute the graph
    input_message = create_question_sentence(target_question_data)
    print ("******** Stage2 input_message **********")
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
        return execute_stage2(expert_info)

    agents_result_dict = mas_result_to_dict(agents_result)

    print ("*********** Stage2 Result **************")
    print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
    print ("****************************************")
    if os.getenv("DATASET") == "egoschema" or os.getenv("DATASET") == "nextqa":
        print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result} (Option{['A', 'B', 'C', 'D', 'E'][prediction_result]})" if 0 <= prediction_result <= 4 else "Error: Invalid result_data value")
    elif os.getenv("DATASET") == "momaqa":
        print (f"Truth: {target_question_data['truth']}, Pred: {prediction_result}")
    print ("****************************************")

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
    os.environ["VIDEO_FILE_NAME"] = "f0444a6e-43be-4ed8-8d50-9acf95da4006"
    data = {
        "ExpertName1":"Film Studies Expert", 
        "ExpertName1Prompt":"You are a Film Studies Expert. Step 1: Understand the purpose by analyzing the question about the character's primary objective in the video. Pay close attention to the narrative structure, character development, and visual storytelling elements. Evaluate the significance of each option based on how it aligns with common themes and objectives in film narratives. Step 2: Gather insights from your knowledge of film theory, narrative techniques, and character analysis, as well as any relevant insights from other experts. Step 3: Critically assess each option, considering how it fits within the context of the video’s narrative and visual cues, and rank them based on their narrative coherence. Incorporate insights from other experts to enhance your analysis. Step 4: Conclude with two detailed interpretations of the character's primary objective, explaining your reasoning with examples from the video and reflecting on any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool.",
        "ExpertName2":"Cognitive Psychologist", 
        "ExpertName2Prompt":"You are a Cognitive Psychologist. Step 1: Understand the purpose by analyzing the question about the character's primary objective in the video. Focus on the character's behavior, decision-making processes, and interactions with objects. Evaluate the significance of each option based on cognitive theories of goal-directed behavior and motivation. Step 2: Gather insights from cognitive psychology, including theories of attention, perception, and motivation, as well as reflections from other experts if mentioned. Step 3: Reflect on the character's actions and choices, logically ranking each option based on its alignment with cognitive principles of goal-oriented behavior. Consider insights from other experts to provide a more comprehensive analysis. Step 4: Conclude with two detailed explanations of the character's primary objective, supported by cognitive theories and examples from the video, while incorporating any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool.",
        "ExpertName3":"Behavioral Analyst",
        "ExpertName3Prompt":"You are a Behavioral Analyst. Step 1: Understand the purpose by analyzing the question about the character's primary objective in the video. Focus on observable behaviors, patterns, and routines. Evaluate the significance of each option based on behavioral theories and the consistency of actions shown in the video. Step 2: Gather insights from behavioral analysis, including reinforcement, habit formation, and routine behaviors, as well as any relevant insights from other experts. Step 3: Reflect on the character's behavior patterns, logically ranking each option based on its consistency with behavioral principles. Incorporate insights from other experts to enhance your analysis. Step 4: Conclude with two detailed interpretations of the character's primary objective, explaining your reasoning with examples from the video and reflecting on any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool.",
    }
    # data = {
    # "ExpertName1": "Art and Design Specialist",
    # "ExpertName1Prompt": "You are an expert in Art and Design. Your role is to analyze the manipulation of cotton wool and its potential artistic applications. Step 1: Understand the purpose by analyzing the question about manipulating cotton wool and the potential artistic outcomes. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each manipulation technique discussed, considering both visual cues and contextual information. Step 2: Gather insights from your professional experience related to art creation, including techniques of sculpture, textile art, and design, as well as reflections from other experts if mentioned. Step 3: Think critically about each of the manipulation techniques in relation to artistic creation, logically ranking each option in detail based on its importance while considering insights shared by other specialists. Step 4: Conclude with two detailed reasons why these manipulations could be used in art, explaining your reasoning with examples and reflecting any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool.",

    # "ExpertName2": "Textile Engineer",
    # "ExpertName2Prompt": "You are an expert in Textile Engineering. Your task is to analyze the manipulation of cotton wool and its potential applications in textile engineering. Step 1: Understand the purpose by analyzing the question about manipulating cotton wool and its potential applications in textile engineering. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each manipulation technique discussed, considering both visual cues and contextual information. Step 2: Gather insights from your professional experience related to textile production, including the properties of cotton wool, garment creation, and material manipulation, as well as reflections from other experts if mentioned. Step 3: Think critically about each of the manipulation techniques in relation to textile engineering, logically ranking each option in detail based on its importance while considering insights shared by other specialists. Step 4: Conclude with two detailed reasons why these manipulations could be used in textile engineering, explaining your reasoning with examples and reflecting any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool.",

    # "ExpertName3": "Occupational Therapist",
    # "ExpertName3Prompt": "You are an expert in Occupational Therapy. Your role is to analyze the manipulation of cotton wool and its potential therapeutic benefits. Step 1: Understand the purpose by analyzing the question about manipulating cotton wool and its potential therapeutic benefits. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each manipulation technique discussed, considering both visual cues and contextual information. Step 2: Gather insights from your professional experience related to therapeutic practices, including hand-eye coordination, fine motor skills, and sensory activities, as well as reflections from other experts if mentioned. Step 3: Think critically about each of the manipulation techniques in relation to therapy, logically ranking each option in detail based on its importance while considering insights shared by other specialists. Step 4: Conclude with two detailed reasons why these manipulations could be used in therapy, explaining your reasoning with examples and reflecting any useful points from other agents. Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion.\nBe sure to call the Analyze video tool."
    # }
    # data = {
    # "ExpertName1": "Art Historian",
    # "ExpertName1Prompt": "You are an expert in Art History. Your role is to analyze the painting strategy of C based on the sequence of actions and techniques observed in the video. Step 1: Understand the purpose by analyzing the question about C's painting strategy and the provided video. Focus on identifying historical painting techniques and strategies that might be relevant. Evaluate the significance of each option based on historical context and traditional methods. Step 2: Gather insights from your knowledge of art history, including famous artists' techniques and the evolution of painting strategies over time. Consider insights from other experts if mentioned. Step 3: Reflect on each option, ranking them based on historical accuracy and relevance, while considering insights from other fields. Step 4: Conclude with two detailed explanations of C's strategy, ensuring they are logically derived from historical analysis and incorporating any useful points from other experts.\nBe sure to call the Analyze video tool.",

    # "ExpertName2": "Cognitive Psychologist",
    # "ExpertName2Prompt": "You are an expert in Cognitive Psychology. Your task is to analyze the painting strategy of C based on cognitive processes and decision-making observed in the video. Step 1: Understand the purpose by analyzing the question about C's painting strategy and the provided video. Focus on identifying cognitive processes involved in painting, such as decision-making and problem-solving. Evaluate the significance of each option based on cognitive strategies. Step 2: Gather insights from cognitive psychology, including theories on creativity, attention, and learning. Consider insights from other experts if mentioned. Step 3: Reflect on each option, ranking them based on cognitive efficiency and strategy, while considering insights from other fields. Step 4: Conclude with two detailed explanations of C's strategy, ensuring they are logically derived from cognitive analysis and incorporating any useful points from other experts.\nBe sure to call the Analyze video tool.",

    # "ExpertName3": "Materials Scientist",
    # "ExpertName3Prompt": "You are an expert in Materials Science. Your role is to analyze C's painting strategy based on the materials and tools used in the process. Step 1: Understand the purpose by analyzing the question about C's painting strategy and the provided video. Focus on identifying the materials and tools used in the painting process. Evaluate the significance of each option based on material properties and usage. Step 2: Gather insights from your knowledge of materials science, including the properties of paints, brushes, and other tools. Consider insights from other experts if mentioned. Step 3: Reflect on each option, ranking them based on material efficiency and innovation, while considering insights from other fields. Step 4: Conclude with two detailed explanations of C's strategy, ensuring they are logically derived from material analysis and incorporating any useful points from other experts.\nBe sure to call the Analyze video tool."
    # }

    execute_stage2(data)
