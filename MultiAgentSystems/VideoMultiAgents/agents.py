from typing import TypedDict, Annotated
from single_agent import execute_single_agent
from util import save_result
from util import set_environment_variables

from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.analyze_video_gemini import analyze_video_gemini
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from tools.analyze_all_gpt4o import analyze_all_gpt4o

class VMAState(TypedDict, total=False):
    question: str
    video_id: Annotated[str, "readonly"] 
    dataset: str
    json_data: str
    answer_json_data: str

    video_agent_answer: str
    caption_agent_answer: str
    graph_agent_answer: str
    orchestrator_agent_answer: str
    true_answer: str

    results_file_path: str

def video_agent(state: VMAState) -> VMAState:
    """Analyze the video content (frames/audio) and draft an answer."""
    try:
        print(f"Starting video agent for video_id: {state['video_id']} in dataset: {state['dataset']}")
        
        # Set environment variables for this process
        set_environment_variables(state["dataset"], state["video_id"], state["json_data"], state["answer_json_data"])
        
        tools = [analyze_video_gemini]
        print("Running video agent tools...")
        result, agent_response, agent_prompts, tool_calls = execute_single_agent(tools)
        
        save_result(state["results_file_path"], "video-agent", state["video_id"], agent_prompts, 
                    agent_response, result, tool_calls, save_backup=False)
        
        print(f"Video agent completed for video_id: {state['video_id']}")
        state["video_agent_answer"] = agent_response
        return state
    except Exception as e:
        print(f"Error in video agent for video_id: {state['video_id']}: {e}")
        raise

def caption_agent(state: VMAState) -> VMAState:
    """Leverage captions/transcript to draft an answer."""
    try:
        print(f"Starting caption agent for video_id: {state['video_id']} in dataset: {state['dataset']}")
        
        # Set environment variables for this process
        set_environment_variables(state["dataset"], state["video_id"], state["json_data"], state["answer_json_data"])

        tools = [retrieve_video_clip_captions]
        print("Running caption agent tools...")
        result, agent_response, agent_prompts, tool_calls = execute_single_agent(tools)
        
        save_result(state["results_file_path"], "caption-agent", state["video_id"], agent_prompts, 
                    agent_response, result, tool_calls, save_backup=False)
        
        print(f"Caption agent completed for video_id: {state['video_id']}")
        state["caption_agent_answer"] = agent_response
        return state
    except Exception as e:
        print(f"Error in caption agent for video_id: {state['video_id']}: {e}")
        raise

def graph_agent(state: VMAState) -> VMAState:
    """Consult a knowledge graph to draft an answer."""
    try:
        print(f"Starting graph agent for video_id: {state['video_id']} in dataset: {state['dataset']}")
        
        # Set environment variables for this process
        set_environment_variables(state["dataset"], state["video_id"], state["json_data"], state["answer_json_data"])

        tools = [retrieve_video_scene_graph]
        print("Running graph agent tools...")
        result, agent_response, agent_prompts, tool_calls = execute_single_agent(tools)
        
        save_result(state["results_file_path"], "graph-agent", state["video_id"], agent_prompts, 
                    agent_response, result, tool_calls, save_backup=False)
        
        #print(f"Graph agent completed for video_id: {state['video_id']}")
        state["graph_agent_answer"] = agent_response
        return state
    except Exception as e:
        print(f"Error in graph agent for video_id: {state['video_id']}: {e}")
        raise

def orchestrator_agent(state: VMAState) -> VMAState:
    """Orchestrate the process across multiple agents."""
    try:
        print(f"Starting orchestrator agent for video_id: {state['video_id']} in dataset: {state['dataset']}")
        
        # Set environment variables for this process
        set_environment_variables(state["dataset"], state["video_id"], state["json_data"], state["answer_json_data"])
        
        print(f"Orchestrator agent completed for video_id: {state['video_id']}")
        return state
    except Exception as e:
        print(f"Error in orchestrator agent for video_id: {state['video_id']}: {e}")
        raise
