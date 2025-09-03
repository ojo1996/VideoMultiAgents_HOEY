import os
import time
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Import utility functions (e.g., for post-processing and question sentence generation)
from util import post_process, create_question_sentence, prepare_intermediate_steps

# Retrieve the Gemini API key from the environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Instantiate the Gemini LLM
llm_gemini = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=False
)

def create_agent(llm, tools: list, system_prompt: str):
    """
    Create a Gemini agent with the given system prompt and tools.
    Gemini does not support system messages, so merge the system prompt into the first human message.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_executor(question_sentence):
        # Merge system prompt and question into one HumanMessage
        message = HumanMessage(content=f"{system_prompt}\n\n{question_sentence}")
        # Gemini expects a list of messages
        return llm_with_tools.invoke([message])

    return agent_executor

def execute_single_agent(tools, use_summary_info=False):
    """
    Execute the VideoQuestionAnswering task using a single agent.

    The agent's task is to analyze the video using the available tools and select the most plausible answer
    among the five options provided.
    """
    # Load the question and answer data from an environment variable
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    video_id = os.environ["VIDEO_FILE_NAME"]
    answers = json.loads(os.getenv("ANSWERS_JSON_STR"))

    if video_id not in answers.keys():
        #print(f"Skipping: {video_id} because it lacks an answer.")
        raise ValueError(f"Video {video_id} does not have an answer in the answers file.")

    # Create a system prompt that outlines the task instructions only
    system_prompt = (
        "Your task is to perform Video Question Answering. Analyze the video using the available tools, "
        "carefully reasoning through each step. Then, select the most plausible answer from the five given options. "
        "Finally, respond with 'FINISH' followed by your final answer, which should be one of the following: "
        "'Option A', 'Option B', 'Option C', 'Option D', or 'Option E'."
    )

    if use_summary_info:
        summary_info = json.loads(os.getenv("SUMMARY_INFO"))
        system_prompt += "\n\n[Video Summary Information]\n"
        system_prompt += "Entire Summary: \n" + summary_info["entire_summary"] + "\n\n"
        system_prompt += "Detail Summaries: \n" + summary_info["detail_summaries"]

    # Generate the question sentence using the provided utility (do not include this in the system prompt)
    question_sentence = create_question_sentence(target_question_data)

    # Create the single agent with the defined system prompt and tools
    single_agent = create_agent(llm_gemini, tools, system_prompt=system_prompt)

    # Print the input message for debugging purposes
    #print("******** Single Agent Input Message **********")
    #print(question_sentence)
    #print("*****************************************************")

    # Call the agent function directly
    result = single_agent(question_sentence)
    output_content = result.content 

    # Process the output result (e.g., converting answer to expected format)
    prediction_result = post_process(output_content)

    # If the result is invalid, retry the task
    if prediction_result == -1:
        #print("***********************************************************")
        #print("Error: The result is -1. Retrying VideoQuestionAnswering with the single agent.")
        #print("***********************************************************")
        time.sleep(1)
        return execute_single_agent(tools)

    # Print the result for debugging purposes
    #print("*********** Single Agent Result **************")
    #print(output_content)
    #print("******************************************************")

    # Display truth and prediction if a dataset is specified via environment variable
    if os.getenv("DATASET") in ["egoschema", "nextqa", "intentqa", "hourvideo"]:
        if 0 <= prediction_result <= 4:
            print(
                f"Truth: {answers[video_id]}, "
                f"Pred: {prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})"
            )
        else:
            print("Error: Invalid prediction result value")
    elif os.getenv("DATASET") == "momaqa":
        print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result}")
    print("******************************************************")

    # Build additional outputs for debugging and traceability
    print(result.tool_calls)

    agents_result_dict = {"output": output_content, "tool_calls": result.tool_calls}
    agent_prompts = {"system_prompt": system_prompt}

    return prediction_result, agents_result_dict, agent_prompts, result.tool_calls

if __name__ == "__main__":
    execute_single_agent([])
