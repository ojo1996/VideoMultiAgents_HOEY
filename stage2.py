import os
import time
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Import required tools for video analysis
from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.analyze_video_gpt4o import analyze_video_gpt4o
from tools.retrieve_video_clip_captions_with_graph_data import retrieve_video_clip_captions_with_graph_data
from tools.analyze_video_gpt4o_with_videotree_frame_sampling import analyze_video_gpt4o_with_videotree_frame_sampling
from tools.retrieve_video_scene_graphs_and_enriched_captions import retrieve_video_scene_graphs_and_enriched_captions
# Import utility functions (e.g., for post-processing and question sentence generation)
from util import post_process, create_question_sentence

# Retrieve the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the list of tools for video analysis
tools = [analyze_video_gpt4o]
# tools = [analyze_video_gpt4o_with_videotree_frame_sampling]
# tools = [retrieve_video_clip_captions]
# tools = [retrieve_video_clip_captions_with_graph_data]
# tools = [retrieve_video_scene_graphs_and_enriched_captions]
# tools = [analyze_video_gpt4o_with_videotree_frame_sampling, retrieve_video_clip_captions]

# Instantiate the LLM with appropriate configurations
llm_openai = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-4o',
    temperature=0.7,
    disable_streaming=True
)

def create_agent(llm, tools: list, system_prompt: str):
    """
    Create an agent with the given system prompt and tools.
    The prompt contains placeholders for the conversation and the agent's internal reasoning (scratchpad).
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def execute_video_question_answering():
    """
    Execute the VideoQuestionAnswering task using a single enhanced agent.

    The agent's task is to analyze the video using available tools and select the most plausible answer
    among the five options provided.

    The system prompt instructs the agent to always perform a self-reflection on its reasoning before 
    outputting the final answer.

    This version does not use an external iterative prompting loop; instead, it performs a single invocation.

    The function returns a 3-tuple:
        (prediction_result, agents_result_dict, agent_prompts)
    """
    # Load the question data from an environment variable
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))

    # Enhanced system prompt with chain-of-thought instructions and self-reflection requirement

    system_prompt = (
        "Your task is VideoQuestionAnswering. You must analyze the video using available tools and generate "
        "a concise answer (1-2 words) to the given open-ended question.\n"
        "\n"
        "To ensure a high-quality response, evaluate your answer based on:\n"
        "1. **Evidence Support:** Assess the extent to which direct evidence from the video supports your answer.\n"
        "\n"
        "Follow these steps:\n"
        "a. Extract and summarize key details from the video, focusing on relevant actions, objects, and interactions.\n"
        "b. Determine the most plausible 1-2 word answer based only on the extracted evidence.\n"
        "c. Justify your choice using direct evidence from the video.\n"
        "d. Record your step-by-step chain-of-thought and intermediate reasoning in the scratchpad.\n"
        "e. ALWAYS include a 'Self-Reflection:' section before your final answer where you review your reasoning, check for inconsistencies, and refine your response.\n"
        "f. Finally, output your final answer in the following format: **'Pred: xxxxx'** (1-2 words only).\n"
        "\n"
        "--- Example ---\n"
        "Question: What is the chef holding while cutting vegetables?\n"
        "\n"
        "Chain-of-Thought:\n"
        "  - The video shows the chef standing at a counter, using a knife to cut vegetables.\n"
        "  - The chef holds the vegetables steady with their left hand while cutting with their right hand.\n"
        "  - Evidence Support: The video clearly shows the chef gripping a vegetable while cutting.\n"
        "\n"
        "Self-Reflection:\n"
        "  - The extracted details strongly support the answer 'vegetable'. There are no conflicting observations.\n"
        "\n"
        "Pred: vegetable\n"
        "--- End Example ---\n\n"
        "Now, please analyze the provided video, reflect on your reasoning using **only direct evidence from the video**, "
        "and generate the most well-supported 1-2 word answer in the format: **'Pred: xxxxx'**."
    )

    # Generate the question sentence using the provided utility (this text is not part of the system prompt)
    question_sentence = create_question_sentence(target_question_data)

    # Create the single agent with the defined enhanced system prompt and tools
    single_agent = create_agent(llm_openai, tools, system_prompt=system_prompt)

    # Create the input state message with the question sentence as user input
    state = {"messages": [HumanMessage(content=question_sentence, name="user")]}
    result = single_agent.invoke(state)
    output_content = result["output"]

    # Process the output result (e.g., converting the answer to the expected format)
    prediction_result = post_process(output_content)

    # If the result is invalid, retry the task
    if prediction_result == -1:
        print("***********************************************************")
        print("Error: The result is -1. Retrying VideoQuestionAnswering with the single agent.")
        print("***********************************************************")
        time.sleep(1)
        return execute_video_question_answering()

    # Print the final result for debugging purposes
    print("*********** Final Agent Result **************")
    print(output_content)
    print("**********************************************")

    # Display truth and prediction if a dataset is specified via environment variable
    if os.getenv("DATASET") in ["egoschema", "nextqa"]:
        if 0 <= prediction_result <= 4:
            print(
                f"Truth: {target_question_data['truth']}, "
                f"Pred: {prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})"
            )
        else:
            print("Error: Invalid prediction result value")
    elif os.getenv("DATASET") == "momaqa":
        print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result}")
    print("**********************************************")

    # Build additional outputs for debugging and traceability
    agents_result_dict = {"single_agent": output_content}
    agent_prompts = {"system_prompt": system_prompt}

    return prediction_result, agents_result_dict, agent_prompts

if __name__ == "__main__":
    execute_video_question_answering()
