import os
import time
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Import utility functions (e.g., for post-processing and question sentence generation)
from util import post_process, create_question_sentence, prepare_intermediate_steps

# Retrieve the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

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
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
    return executor

def execute_single_agent(tools):
    """
    Execute the VideoQuestionAnswering task using a single agent.

    The agent's task is to analyze the video using available tools and select the most plausible answer
    among the five options provided.
    """
    # Load the question data from an environment variable
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))

    # Create a system prompt that outlines the task instructions only
    system_prompt = (
        "Your task is VideoQuestionAnswering. You must analyze the video using available tools and choose "
        "the most plausible answer among the five options provided. Think step by step and eventually respond "
        "with 'FINISH' followed by your final answer."
    )

    summary_info = json.loads(os.getenv("SUMMARY_INFO"))
    system_prompt += "\n\n[Video Summary Information]\n"
    system_prompt += "Entire Summary: \n" + summary_info["entire_summary"] + "\n\n"
    system_prompt += "Detail Summaries: \n" + summary_info["detail_summaries"]

    # Generate the question sentence using the provided utility (do not include this in the system prompt)
    question_sentence = create_question_sentence(target_question_data)

    # Create the single agent with the defined system prompt and tools
    single_agent = create_agent(llm_openai, tools, system_prompt=system_prompt)

    # Print the input message for debugging purposes
    print("******** Single Agent Input Message **********")
    print(question_sentence)
    print("*****************************************************")

    # Create the input state message with the question sentence
    state = {"messages": [HumanMessage(content=question_sentence, name="system")]}
    result = single_agent.invoke(state)
    output_content = result["output"]

    # Process the output result (e.g., converting answer to expected format)
    prediction_result = post_process(output_content)

    # If the result is invalid, retry the task
    if prediction_result == -1:
        print("***********************************************************")
        print("Error: The result is -1. Retrying VideoQuestionAnswering with the single agent.")
        print("***********************************************************")
        time.sleep(1)
        return execute_single_agent(tools)

    # Print the result for debugging purposes
    print("*********** Single Agent Result **************")
    print(output_content)
    print("******************************************************")

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
    print("******************************************************")

    # Build additional outputs for debugging and traceability
    intermediates = prepare_intermediate_steps(result.get("intermediate_steps", []))
    agents_result_dict = {"output": output_content, "intermediate_steps": intermediates}
    agent_prompts = {"system_prompt": system_prompt}

    return prediction_result, agents_result_dict, agent_prompts

if __name__ == "__main__":
    execute_single_agent([])
