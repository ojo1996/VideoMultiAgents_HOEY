import os
import re
import json
import base64
import numpy as np
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate


def encode_image_to_data_uri(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"


def get_image_paths(directory, limit=10):
    supported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    all_files = sorted([
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(supported_extensions)
    ])

    if not all_files:
        return []

    # If the limit is 1, return the first file
    if limit == 1:
        return [all_files[0]]

    # Uniformly sample the files
    indices = np.linspace(0, len(all_files) - 1, limit, dtype=int)
    sampled_files = [all_files[i] for i in indices]

    return sampled_files


class ExpertResponse(BaseModel):
    ExpertName1:       str = Field(..., description="Name of Expert 1's field of expertise")
    ExpertName1Prompt: str = Field(..., description="Prompt for Expert 1 with clear instructions following the four-step reasoning structure")
    ExpertName2:       str = Field(..., description="Name of Expert 2's field of expertise")
    ExpertName2Prompt: str = Field(..., description="Prompt for Expert 2 with clear instructions following the four-step reasoning structure")
    ExpertName3:       str = Field(..., description="Name of Expert 3's field of expertise")
    ExpertName3Prompt: str = Field(..., description="Prompt for Expert 3 with clear instructions following the four-step reasoning structure")


# prompt template for stage1
common_instructions_template = """
[Instructions]
Your task is to identify three experts from different fields to answer a given question related to the provided video.
You must also provide specific prompts for each expert, instructing them to answer the questions from their unique perspectives using a four-step structured reasoning approach.
Additionally, each expert should consider the responses and insights of other agents during their reasoning to form a more holistic answer without losing the independence of their own perspective.

- Ensure each expert represents a completely distinct area of expertise to provide diverse viewpoints and maintain independence in their own analysis.
- Each expert prompt should also guide the agent to incorporate, reflect, and reason upon insights from other experts.
- Create detailed prompts for each expert, explicitly breaking down the four-step thought process each needs to follow in addressing the question and justifying their conclusions while considering other agents' responses.

### Steps
1. **Understanding the Purpose**: Understand the purpose of the question by analyzing what is being asked and identifying the criteria that must be used to evaluate each selected answer option.
2. **Gathering Information**: Gather all necessary information. Use available tools, data, prior knowledge from the expert's domain, and insights provided by other experts to collect key information to solve the problem.
3. **Reflecting**: Reflect deeply on the gathered information, ensuring logical coherence. Evaluate each option thoroughly and rank them according to their relevance based on the expert's specific field of knowledge. Incorporate insights or useful points from other experts while keeping the main reasoning within the limits of the specific field of expertise.
4. **Concluding**: Conclude with two detailed and plausible answers, explaining each conclusion and ensuring they are derived logically from the analysis conducted in the prior steps.

### Example prompt for ExpertName1Prompt
You are a Housekeeping Expert. Step 1: Understand the purpose by analyzing the question about maintaining a clean home environment and any provided video. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each aspect discussed, considering both visual cues and contextual information. Step 2: Gather insights from your professional experience related to household management, including benefits of hygiene, organization, and overall housekeeping, as well as reflections from other experts if mentioned. Step 3: Think critically about each of the aspects in relation to cleanliness, logically ranking each option in detail based on its importance while considering insights shared by other specialists. Step 4: Conclude with two detailed reasons why cleanliness is essential, explaining your reasoning with examples and reflecting any useful points from other agents.Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion. Note:Be sure to use tool to guide your answer!
### Example prompt for ExpertName2Prompt
You are a Psychologist. Step 1: Understand the purpose by analyzing the question about maintaining a clean home environment and any provided video. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each aspect discussed, considering both visual cues and contextual information. Step 2: Gather relevant theories and research findings on the correlation between cleanliness and improved mental health, as well as reviewing relevant insights from other experts. Step 3: Think critically about factors like reduced stress and improved organization, carefully ranking them while considering any useful points mentioned by others. Step 4: Conclude by providing two comprehensive answers, each reflecting different psychological dimensions of cleanliness, and also mentioning useful reflections from other experts when relevant.Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion. Note:Be sure to use tool to guide your answer!
### Example prompt for ExpertName3Prompt
You are a Pest Control Specialist. Step 1: Understand the purpose by analyzing the question about maintaining a clean home environment and any provided video. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each aspect discussed, considering both visual cues and contextual information. Step 2: Gather information from your background on effective pest reduction methods, linking cleanliness to pest prevention. Evaluate any relevant points that other experts might have highlighted. Step 3: Critically assess the given options based on the potential to reduce pests, incorporate useful reflections from other fields for a more holistic argument, and rank them accordingly. Step 4: Conclude with two well-founded answers, backed by evidence of how cleanliness and pest avoidance are interrelated, detailing your reasoning behind each suggestion, taking into consideration any other agent insights where valuable.Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion. Note:Be sure to use tool to guide your answer!

### Notes
- Select experts whose fields provide distinct and non-overlapping perspectives to ensure diverse, valuable insights.
- Explanations must clearly adhere to the professional background of each expert while also referencing and considering other agents’ insights where they enhance the quality of the response.
- For each expert, make sure to articulate the step-by-step reasoning and include reflections on insights from other agents before stating the conclusions.
- The JSON format must be exactly adhered to for uniformity in the output.

"""

# for egoschema and nqxt-qa dataset
five_choice_template = """
[Question and 5 Options to Solve]
Question: {question}
Option A: {option_1}
Option B: {option_2}
Option C: {option_3}
Option D: {option_4}
Option E: {option_5}

""" + common_instructions_template

# for moma-qa dataset
open_question_template = """
[Question to Solve]
Question: {question}

""" + common_instructions_template


def execute_stage1():

    image_directory         = os.getenv("IMAGES_DIR_PATH")
    frame_num               = int(os.getenv("FRAME_NUM"))
    video_filename          = os.getenv("VIDEO_FILE_NAME")
    qa_json_str             = os.getenv("QA_JSON_STR")

    # Convert the question data keys to match the expected format
    question_data_raw = json.loads(qa_json_str)
    question_data = {}
    for key, value in question_data_raw.items():
        new_key = re.sub(r"option (\d+)", lambda m: f"option_{int(m.group(1))+1}", key)
        question_data[new_key] = value

    # Create data uri for each images
    target_directory = os.path.join(image_directory, video_filename)
    image_paths      = get_image_paths(target_directory, frame_num)
    image_data_uris  = [encode_image_to_data_uri(path) for path in image_paths]

    # prepare the chat prompt
    if "option_1" in question_data.keys():
        question_template = five_choice_template
    else:
        question_template = open_question_template

    system_message = SystemMessagePromptTemplate.from_template("You are a skilled prompt editor. Write the optimal prompts for Agents to solve the Video Question Answering problems that the user inputs.")
    human_message = HumanMessagePromptTemplate.from_template(question_template)
    image_messages = [HumanMessage(content="", additional_kwargs={"image_url": data_uri}) for data_uri in image_data_uris]
    chat_prompt    = ChatPromptTemplate.from_messages([system_message, human_message] + image_messages)

    # Define the chat model and chain
    chat  = ChatOpenAI(temperature=0, model_name="gpt-4o-2024-08-06")
    chain = chat_prompt | chat.with_structured_output(ExpertResponse)

    # Execute the chain
    expert_info = chain.invoke(question_data)

    # Convert the expert_info object to dictionary
    expert_info_dict = expert_info.model_dump()

    # replace newline characters to prevent errors in json serialization
    expert_info_dict["ExpertName1Prompt"] = expert_info_dict["ExpertName1Prompt"].replace('\n',' ')
    expert_info_dict["ExpertName2Prompt"] = expert_info_dict["ExpertName2Prompt"].replace('\n',' ')
    expert_info_dict["ExpertName3Prompt"] = expert_info_dict["ExpertName3Prompt"].replace('\n',' ')

    print ("*********** Stage1 Result **************")
    print(json.dumps(expert_info_dict, indent=2, ensure_ascii=False))
    print ("****************************************")

    return expert_info_dict


if __name__ == "__main__":

    # Example environment variables
    os.environ["OPENAI_API_KEY"] = "sk-xxxxxxx"
    os.environ["IMAGES_DIR_PATH"] = "./images/"
    os.environ["FRAME_NUM"] = "90"
    os.environ["VIDEO_FILE_NAME"] = "0a01d7d0-11d6-4af6-abd9-2025656d3c63"
    os.environ["QA_JSON_STR"] = '{"question": "What is the importance of drinking tea?", "option 0": "To stay hydrated", "option 1": "To socialize", "option 2": "To take breaks", "option 3": "To mark progress", "option 4": "To relax"}'

    execute_stage1()
