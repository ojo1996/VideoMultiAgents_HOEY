import os
import time
import requests
import datetime
from retry import retry
from openai import OpenAI
import re
import json
import random
import glob
import base64
from filelock import FileLock
from mimetypes import guess_type
import numpy as np
from typing import Literal
from google import genai
from google.genai import types
from icecream import ic

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


@retry(tries=3, delay=3)
def ask_gpt4_omni(openai_api_key="", prompt_text="", temperature=0.0, image_dir="", vid="", frame_num=18, detail="low", use_selected_images=None, json_schema=None):
    model_name = "gpt-4o"

    client = OpenAI(api_key=openai_api_key)

    if image_dir != "" and vid != "":
        frame_path_list = sorted(glob.glob(os.path.join(image_dir, vid, "*")))
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        frame_path_list = [path for path in frame_path_list if os.path.splitext(path)[1].lower() in valid_extensions]

        frames = []
        if use_selected_images is not None:
            for image_path in use_selected_images:
                data_url = local_image_to_data_url(image_path)
                frames.append({ "type": "image_url", "image_url": { "url": data_url, "detail": detail } })
        else:
            if len(frame_path_list) <= frame_num:
                selected_paths = frame_path_list
            else:
                # uniformly sample frames
                indices = [int(round(x)) for x in np.linspace(0, len(frame_path_list) - 1, frame_num)]
                selected_paths = [frame_path_list[i] for i in indices]

            for image_path in selected_paths:
                data_url = local_image_to_data_url(image_path)
                frames.append({ "type": "image_url", "image_url": { "url": data_url, "detail": detail } })

        print(f'Sending {len(frames)} {detail}-detail frames')

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                { "role": "system", "content": "You are a helpful expert in first person view video analysis." },
                { "role": "user", "content": prompt_text },
                { "role": "user", "content": frames }
            ],
            max_tokens=3000,
            temperature=temperature,
            response_format=json_schema
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": prompt_text }
            ],
            max_tokens=3000,
            temperature=temperature,
            response_format=json_schema
        )

    print(f"ask_gpt4_omni: prompt_tokens={response.usage.prompt_tokens}, completion_tokens={response.usage.completion_tokens}, total_tokens={response.usage.total_tokens}")

    return response.choices[0].message.content


# @retry(tries=3, delay=3)
def ask_gemini(prompt_text: str, video_path: str = "", temperature=0.7) -> str:
    """
    Send a prompt to Gemini 2.0 for video analysis.

    Args:
        prompt_text (str): The prompt text to send to Gemini
        video_path (str): Path to the video file to analyze

    Returns:
        str: Gemini's response text
    """
    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Use Gemini 2.0 Flash model optimized for video
    model_name = "gemini-2.0-flash"

    generation_config = types.GenerateContentConfig(
        max_output_tokens=3000,
        temperature=temperature,
        seed=42,
        safety_settings= [
          types.SafetySetting(
              category='HARM_CATEGORY_HATE_SPEECH',
              threshold='BLOCK_NONE'
          ),
          types.SafetySetting(
              category='HARM_CATEGORY_HARASSMENT',
              threshold='BLOCK_NONE'
          ),
          types.SafetySetting(
              category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
              threshold='BLOCK_NONE'
          ),
          types.SafetySetting(
              category='HARM_CATEGORY_DANGEROUS_CONTENT',
              threshold='BLOCK_NONE'
          ),
          types.SafetySetting(
              category='HARM_CATEGORY_CIVIC_INTEGRITY',
              threshold='BLOCK_NONE'
          ),
      ]
    )
    if video_path:
        # Check if video is already in cache
        cache_file_path = f"gemini_video_cache.json"
        video_key = os.path.basename(os.path.splitext(video_path)[0])
        video_file_name = None
        
        # Use FileLock to safely read from cache
        lock = FileLock(f"{cache_file_path}.lock")
        
        with lock:
            # Load cache if exists
            if os.path.exists(cache_file_path):
                try:
                    with open(cache_file_path, 'r') as f:
                        video_cache = json.load(f)
                    if video_key in video_cache:
                        video_file_name = video_cache[video_key]
                    elif len(video_cache) >= 1500: # need to delete old video
                        # Delete all files in cache to free up space
                        print("Cache limit reached. Clearing video cache...")
                        # Get all video file names from the cache
                        video_files_to_delete = list(video_cache.values())
                        import concurrent.futures
                        # Delete files in parallel using ThreadPoolExecutor
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            # Submit delete tasks for each file
                            delete_futures = [
                                executor.submit(client.files.delete, name=file_name)
                                for file_name in video_files_to_delete
                            ]
                            
                            # Wait for all deletions to complete
                            concurrent.futures.wait(delete_futures)
                        
                        # Clear the cache dictionary
                        video_cache = {}
                        # Write empty cache to file
                        with open(cache_file_path, 'w') as f:
                            json.dump(video_cache, f)
                        print("Cache cleared.")
                except json.JSONDecodeError:
                    video_cache = {}
            else:
                video_cache = {}
        
        # Try to get existing file or upload new one
        if video_file_name:
            try:
                video_file = client.files.get(name=video_file_name)
                print(f'Found cached video: {video_file.name}')
            except Exception as e:
                print(f'Error retrieving cached video: {e}')
                video_file_name = None
        
        if not video_file_name:
            # Upload video file to Gemini
            video_file = client.files.upload(file=video_path)
            
            # Update cache with new mapping using FileLock
            with lock:
                # Re-read the cache in case it was modified by another process
                if os.path.exists(cache_file_path):
                    try:
                        with open(cache_file_path, 'r') as f:
                            video_cache = json.load(f)
                    except json.JSONDecodeError:
                        video_cache = {}
                
                video_cache[video_key] = video_file.name
                with open(cache_file_path, 'w') as f:
                    json.dump(video_cache, f)
                print(f'Added video to cache: {video_key} -> {video_file.name}')

        # Wait for video processing to complete
        while video_file.state == "PROCESSING":
            print('Waiting for video to be processed.')
            time.sleep(10)
            video_file = client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            raise ValueError(video_file.state)
        print(f'Video processing complete: {video_file.uri}')

        # Generate response with video
        response = client.models.generate_content(
            model=model_name,
            contents=[
                video_file,
                prompt_text,
            ],
            config=generation_config
        )
    else:
        # Generate response without video
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt_text],
            config=generation_config
        )

    print(f"ask_gemini: prompt_tokens={response.usage_metadata.prompt_token_count}, completion_tokens={response.usage_metadata.candidates_token_count}, total_tokens={response.usage_metadata.total_token_count}")

    return response.text


def create_mas_stage1_prompt(json_data):
    try:
        question = f"Question: {json_data['question']}"
    except KeyError:
        raise ValueError("The key 'question' is missing in the provided json_data.")

    options = []
    for i in range(5):
        key = f"option {i}"
        if key in json_data:
            # chr(65 + i) represent 'A', 'B', 'C', 'D', 'E'
            options.append(f"Option {chr(65 + i)}: {json_data[key]}")
        else:
            raise ValueError(f"The key '{key}' is missing in the provided json_data.")

    options_str = "\n".join(options)

    # prompt = (
    #     "[Question and 5 Options to Solve]\n"
    #     f"{question}\n"
    #     f"{options_str}\n\n"
    #     "[Instructions]\n"
    #     "Please identify two experts to answer questions related to this video. Name the two types of experts and specify their fields of expertise.\n"
    #     "Ensure the expert types come from different fields to provide diverse perspectives.\n"
    #     "Additionally, create a prompt for each expert to answer the questions. Instruct each expert to provide two answers and explanations.\n\n"
    #     "[Example prompt for ExpertNameXPrompt]\n"
    #     "You are a Housekeeping Expert. Watch the video from the perspective of a professional housekeeper and answer the following questions based on your expertise.\n"
    #     "Please think step-by-step.\n\n"
    #     "[Response Format]\n"
    #     "You must respond using this JSON format:\n"
    #     "{\n"
    #     '  "ExpertName1": "xxxx",\n'
    #     '  "ExpertName1Prompt": "xxxx",\n'
    #     '  "ExpertName2": "xxxx",\n'
    #     '  "ExpertName2Prompt": "xxxx"\n'
    #     "}"
    # )

    prompt = (
        "[Question and 5 Options to Solve]\n"
        f"{question}\n"
        f"{options_str}\n\n"
        "[Instructions]\n"
        "Your task is to identify three experts from different fields to answer a given question related to the provided video. You must also provide specific prompts for each expert, instructing them to answer the questions from their unique perspectives using a four-step structured reasoning approach. Additionally, each expert should consider the responses and insights of other agents during their reasoning to form a more holistic answer without losing the independence of their own perspective.\n\n"
        "- Ensure each expert represents a completely distinct area of expertise to provide diverse viewpoints and maintain independence in their own analysis.\n"
        "- Each expert prompt should also guide the agent to incorporate, reflect, and reason upon insights from other experts.\n"
        "- Create detailed prompts for each expert, explicitly breaking down the four-step thought process each needs to follow in addressing the question and justifying their conclusions while considering other agents' responses.\n\n"
        "### Steps\n\n"
        "1. **Understanding the Purpose**: Understand the purpose of the question by analyzing what is being asked and identifying the criteria that must be used to evaluate each selected answer option.\n"
        "2. **Gathering Information**: Gather all necessary information. Use available tools, data, prior knowledge from the expert's domain, and insights provided by other experts to collect key information to solve the problem.\n"
        "3. **Reflecting**: Reflect deeply on the gathered information, ensuring logical coherence. Evaluate each option thoroughly and rank them according to their relevance based on the expert's specific field of knowledge. Incorporate insights or useful points from other experts while keeping the main reasoning within the limits of the specific field of expertise.\n"
        "4. **Concluding**: Conclude with two detailed and plausible answers, explaining each conclusion and ensuring they are derived logically from the analysis conducted in the prior steps.\n\n"
        "### Output Format\n\n"
        "Provide the responses in the following JSON format:\n\n"
        "{\n"
        '  "ExpertName1": "[Name of Expert Field]",\n'
        '  "ExpertName1Prompt": "[Prompt for Expert 1 with clear instructions to follow the four-step structure and incorporating insights from other agents]",\n'
        '  "ExpertName2": "[Name of Expert Field]",\n'
        '  "ExpertName2Prompt": "[Prompt for Expert 2 with clear instructions to follow the four-step structure and incorporating insights from other agents]",\n'
        '  "ExpertName3": "[Name of Expert Field]",\n'
        '  "ExpertName3Prompt": "[Prompt for Expert 3 with clear instructions to follow the four-step structure and incorporating insights from other agents]"\n'
        "}\n\n"
        "### Example prompt for ExpertName1Prompt\n\n"
        "You are a Housekeeping Expert. Step 1: Understand the purpose by analyzing the question about maintaining a clean home environment and any provided video. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each aspect discussed, considering both visual cues and contextual information. Step 2: Gather insights from your professional experience related to household management, including benefits of hygiene, organization, and overall housekeeping, as well as reflections from other experts if mentioned. Step 3: Think critically about each of the aspects in relation to cleanliness, logically ranking each option in detail based on its importance while considering insights shared by other specialists. Step 4: Conclude with two detailed reasons why cleanliness is essential, explaining your reasoning with examples and reflecting any useful points from other agents.Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion. Note:Be sure to use tool to guide your answer!\n\n"
        "### Example prompt for ExpertName2Prompt\n\n"
        "You are a Psychologist. Step 1: Understand the purpose by analyzing the question about maintaining a clean home environment and any provided video. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each aspect discussed, considering both visual cues and contextual information. Step 2: Gather relevant theories and research findings on the correlation between cleanliness and improved mental health, as well as reviewing relevant insights from other experts. Step 3: Think critically about factors like reduced stress and improved organization, carefully ranking them while considering any useful points mentioned by others. Step 4: Conclude by providing two comprehensive answers, each reflecting different psychological dimensions of cleanliness, and also mentioning useful reflections from other experts when relevant.Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion. Note:Be sure to use tool to guide your answer!\n\n"
        "### Example prompt for ExpertName3Prompt\n\n"
        "You are a Pest Control Specialist. Step 1: Understand the purpose by analyzing the question about maintaining a clean home environment and any provided video. Pay close attention to the video, identifying key details and insights relevant to the question. Evaluate the significance of each aspect discussed, considering both visual cues and contextual information. Step 2: Gather information from your background on effective pest reduction methods, linking cleanliness to pest prevention. Evaluate any relevant points that other experts might have highlighted. Step 3: Critically assess the given options based on the potential to reduce pests, incorporate useful reflections from other fields for a more holistic argument, and rank them accordingly. Step 4: Conclude with two well-founded answers, backed by evidence of how cleanliness and pest avoidance are interrelated, detailing your reasoning behind each suggestion, taking into consideration any other agent insights where valuable.Ensure that all intermediate insights, observations, and reasoning processes from each step are clearly output in detail before presenting the final conclusion. Note:Be sure to use tool to guide your answer!\n\n"
        "### Notes\n\n"
        "- Select experts whose fields provide distinct and non-overlapping perspectives to ensure diverse, valuable insights.\n"
        "- Explanations must clearly adhere to the professional background of each expert while also referencing and considering other agents’ insights where they enhance the quality of the response.\n"
        "- For each expert, make sure to articulate the step-by-step reasoning and include reflections on insights from other agents before stating the conclusions.\n"
        "- The JSON format must be exactly adhered to for uniformity in the output.\n"
    )

    return prompt


def create_question_sentence(question_data:dict, shuffle_questions=False):
    # moma-qa
    if os.getenv("DATASET") == "momaqa":
        prompt = "[Question]\n"
        prompt += "Question: " + question_data["question"]
        return prompt
    
    prompt = "[Question and 5 Options to Solve]\n"
    prompt += "Question: " + question_data["question"]
    # Add options
    if shuffle_questions == False:
        prompt += "\nOption A: " + question_data["option 0"]
        prompt += "\nOption B: " + question_data["option 1"]
        prompt += "\nOption C: " + question_data["option 2"]
        prompt += "\nOption D: " + question_data["option 3"]
        prompt += "\nOption E: " + question_data["option 4"]
    else:
        options_order = ["Option A", "Option B", "Option C", "Option D", "Option E"]
        options = ["Option A", "Option B", "Option C", "Option D", "Option E"]
        random.shuffle(options)
        for option in options:
            prompt += "\n・" + option + ": " + question_data["option " + str(options_order.index(option))]
    return prompt


def create_agent_prompt(question_data: dict, agent_type: Literal["text_expert", "video_expert", "graph_expert"] = "text_expert", use_summary_info=False) -> str:
    """
    Creates a prompt for a specific agent type with predefined instructions.

    :param question_data: Dictionary containing the question data.
    :param agent_type: The type of expert agent. Must be one of:
                        - "text_expert"
                        - "video_expert"
                        - "graph_expert"
                        Default is "text_expert".
    :return: Formatted prompt string.
    """
    prompt = create_question_sentence(question_data)

    if use_summary_info:
        summary_info = json.loads(os.getenv("SUMMARY_INFO"))
        # summary_info = json.loads("/root/VideoMultiAgents/nextqa_summary_cache.json")
        prompt += "\n\n[Video Summary Information]\n"
        prompt += "Entire Summary: \n" + summary_info["entire_summary"] + "\n\n"
        prompt += "Detail Summaries: \n" + summary_info["detail_summaries"]

    prompt += "\n\n[Instructions]\n"
    instructions = {
        "text_expert": "Your task is to answer the question related to the video as a Text Expert. Analyze the video from the perspective of a professional text analyst and answer the following questions based on your expertise. You must use the tool to obtain relevant data, and integrate that data into your answer. Please think step-by-step.",
        "video_expert": "Your task is to answer the question related to the video as a Video Expert. Analyze the video from the perspective of a professional video analyst and answer the following questions based on your expertise. You must use the tool to obtain relevant data, and integrate that data into your answer. Please think step-by-step.",
        "graph_expert": "Your task is to answer the question related to the video as a Graph Expert. Analyze the video from the perspective of a professional graph analyst and answer the following questions based on your expertise. You must use the tool to obtain relevant data, and integrate that data into your answer. Please think step-by-step."
    }

    prompt += instructions[agent_type]

    return prompt


def create_organizer_prompt():

    if os.getenv("DATASET") == "egoschema" or os.getenv("DATASET") == "nextqa":
        organizer_prompt = (
            "[Instructions]\n"
            "You are the organizer of a discussion. "
            "Your task is to summarize the opinions of three Agents, determine whether further discussion is necessary, and, if the discussion is already concluded, provide the final decision.\n"
            "Your output should be one of the following options: OptionA, OptionB, OptionC, OptionD, OptionE, along with an explanation.\n"
            "The correct answer is always within these 5 options.\n"
            "Base your decision primarily on a majority vote. If the opinions of the three Agents are divided, initiate a follow-up discussion to reach a consensus.\n\n"

            "[Output Format]\n"
            "Your response should be formatted as follows:\n"
            "- Additional Discussion Needed: [YES/NO]\n"
            "- Pred: OptionX (If additional discussion is needed, provide the current leading candidate.)\n"
            "- Explanation: Provide a detailed explanation, including reasons for requiring additional discussion or the reasoning behind the final decision."
        )
    elif os.getenv("DATASET") == "momaqa":
        organizer_prompt = (
            "[Instructions]\n"
            "You are the organizer of a discussion. "
            "Your task is to summarize the opinions of three Agents, determine whether further discussion is necessary, and, if the discussion is already concluded, provide the final answer.\n"
            "Your output should be a concise and clear answer to the question, along with an explanation.\n"
            "Base your decision primarily on a majority vote. If the opinions of the three Agents are divided, initiate a follow-up discussion to reach a consensus.\n\n"

            "[Output Format]\n"
            "Your response should be formatted as follows:\n"
            "Note: If any part of the output is a number, represent it as a digit (e.g., '1') rather than as a word (e.g., 'one').\n"
            "- Additional Discussion Needed: [YES/NO]\n"
            "- Pred: [Your final answer, stated as a single word or phrase (e.g., 'basketball player', 'preparing food')]\n"
            "- Explanation: Provide a detailed explanation, including reasons for requiring additional discussion or the reasoning behind the final answer."
        )

    return organizer_prompt


def create_stage2_agent_prompt(question_data:dict, generated_expert_prompt="", shuffle_questions=False, use_summary_info=False):
    prompt = create_question_sentence(question_data, shuffle_questions)

    if use_summary_info:
        summary_info = json.loads(os.getenv("SUMMARY_INFO"))
        prompt += "\n\n[Video Summary Information]\n"
        prompt += "Entire Summary: \n" + summary_info["entire_summary"] + "\n\n"
        prompt += "Detail Summaries: \n" + summary_info["detail_summaries"]
    
    prompt += "\n\n[Instructions]\n"
    prompt += generated_expert_prompt
    return prompt


def create_star_organizer_prompt():
    shared_organizer_prompt = (
        "[Instructions]\n"
        "You are the organizer of a discussion between three expert agents analyzing a video. "
        "Your task is to coordinate the discussion by deciding which agent should speak next or when to conclude with a final answer.\n\n"
        
        "The three agents are:\n"
        "- agent1: An expert video analyzer who focuses on video question answering\n"
        "- agent2: An expert video analyzer who focuses on video captions\n"
        "- agent3: An expert video analyzer who focuses on video scene graphs\n\n"
        
        "After each agent speaks, you must provide a comment on the previous agent's response and then decide one of the following:\n"
        "1. Ask a specific agent to provide more information or address a particular aspect (select 'agent1', 'agent2', or 'agent3').\n"
        "2. Conclude the discussion and provide a final answer (select 'FINAL_ANSWER').\n\n"
        "If you choose to ask another agent, you must specify guidance for that agent by telling it what to focus on or what specific information to look for.\n"
        "Be directive about what information is needed or what aspects to investigate. Focus on requesting objective analysis rather than suggesting specific conclusions. Ask for information or analysis without implying expected outcomes.\n"
        "When agents disagree with each other, you must ask them to look for evidence for the opposing side, rather than reaffirming their own opinion.\n\n"
    )
    if os.getenv("DATASET") == "egoschema" or os.getenv("DATASET") == "nextqa":
        organizer_prompt = (
            shared_organizer_prompt +
            "When concluding the discussion, your final answer should be one of the following options: OptionA, OptionB, OptionC, OptionD, OptionE, along with an explanation.\n"
            "Base your decision on a comprehensive analysis of each agent's opinions and the information provided. Evaluate the reasoning behind each response to determine whether the evidence is sufficient for a final decision or if further input is needed from a specific agent."
        )
    elif os.getenv("DATASET") == "momaqa":
        organizer_prompt = (
            shared_organizer_prompt +
            "When concluding the discussion, your final answer should be a concise and clear answer to the question, along with an explanation.\n"
            "Stated the final answer as a single word or phrase (e.g., 'basketball player', 'preparing food'). If any part of the output is a number, represent it as a digit (e.g., '1') rather than as a word (e.g., 'one')."
        )

    return organizer_prompt


def set_environment_variables(dataset:str, video_id:str, qa_json_data:dict, answers_json_data: dict):
    if dataset == "egoschema": index_name = video_id

    if dataset == "egoschema":
        os.environ["VIDEO_FILE_NAME"] = video_id


    os.environ["SUMMARY_INFO"] = json.dumps(get_video_summary(os.getenv("SUMMARY_CACHE_JSON_PATH"), os.getenv("VIDEO_FILE_NAME")))

    os.environ["VIDEO_INDEX"]     = index_name
    os.environ["QA_JSON_STR"]     = json.dumps(qa_json_data)

    os.environ["ANSWERS_JSON_STR"] = json.dumps(answers_json_data)
    
    #print ("{} : {}".format(video_id, index_name))


def post_process(message:str):

    if os.getenv("DATASET") == "egoschema" or os.getenv("DATASET") == "nextqa" or os.getenv("DATASET") == "intentqa" or os.getenv("DATASET") == "hourvideo":
        prediction_num = post_process_5choice(message)
        if prediction_num == -1:
            prompt = message + "\n\nPlease retrieve the final answer from the sentence above. Your response should be one of the following options: Option A, Option B, Option C, Option D, Option E."
            response_data = ask_gemini(prompt)
            prediction_num = post_process(response_data)
        # print ("prediction_num:", prediction_num)
        return prediction_num
    elif os.getenv("DATASET") == "momaqa":
        prompt = message + "\n\nExtract and output only the content that immediately follows \"- Pred:\" on its line. Do not include any additional text or formatting."
        response_data = ask_gpt4_omni(openai_api_key=os.getenv("OPENAI_API_KEY"), prompt_text=prompt)
        return response_data
    else:
        print ("*************** Invalid dataset ***************")
        return -2


def post_intermediate_process(message:str):
    if os.getenv("DATASET") == "egoschema" or os.getenv("DATASET") == "nextqa":
        prediction_num = post_process_5choice(message)
        if prediction_num == -1:
            prompt = message + "\n\nPlease retrieve the answer from the sentence above. Your response should be one of the following options: Option A, Option B, Option C, Option D, Option E."
            response_data = ask_gpt4_omni(openai_api_key=os.getenv("OPENAI_API_KEY"), prompt_text=prompt)
            prediction_num = post_process(response_data)
        return prediction_num
    elif os.getenv("DATASET") == "momaqa":
        prompt = message + "\n\nExtract and output only the content that immediately follows \"- Pred:\" on its line. Do not include any additional text or formatting."
        response_data = ask_gpt4_omni(openai_api_key=os.getenv("OPENAI_API_KEY"), prompt_text=prompt)
        return response_data


def post_process_5choice(response):
    response = response.lower()
    option_patterns = {
        "option a": 0,
        "option b": 1,
        "option c": 2,
        "option d": 3,
        "option e": 4,
    }

    found_options = []
    for option, value in option_patterns.items():
        # Consider both separated and concatenated patterns
        if re.search(rf'\b{option}\b', response) or re.search(rf'\b{option.replace(" ", "")}\b', response):
            found_options.append(value)

    if len(found_options) == 1:
        return found_options[0]
    else: # If multiple or no options are found, return -1
        return -1


def read_json_file(file_path):
    # print ("read_json_file")
    lock = FileLock(file_path + '.lock')
    try:
        with lock, open(file_path, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(e)
        time.sleep(1)
        read_json_file(file_path)


def select_data_and_mark_as_processing(file_path):
    print ("select_data_and_mark_as_processing")
    dict_data = read_json_file(file_path)

    for i, (video_id, json_data) in enumerate(dict_data.items()):

        if "pred" not in json_data.keys():
            dict_data[video_id]["pred"] = -2
            lock = FileLock(file_path + '.lock')
            with lock, open(file_path, "w") as f:
                json.dump(dict_data, f, indent=4)
            return video_id, json_data
    return None, None

# def select_data_and_mark_as_processing(file_path):
#     print ("select_data_and_mark_as_processing")
#     dict_data = read_json_file(file_path)
#     true_count = 0
#     valid_count = 0
#     for i, (video_id, json_data) in enumerate(dict_data.items()):

#         if "pred" not in json_data.keys():
#             dict_data[video_id]["pred"] = -2
#             with open(file_path, "w") as f:
#                 portalocker.lock(f, portalocker.LOCK_EX)
#                 json.dump(dict_data, f, indent=4)
#                 portalocker.unlock(f)
#             accuracy = (true_count/valid_count*100) if valid_count > 0 else 0
#             print(f"Accuracy till Question no. {i}:========================================= {accuracy:.2f}%")
#             return video_id, json_data

#         if dict_data[video_id]["pred"] == -2:
#             accuracy = (true_count/valid_count*100) if valid_count > 0 else 0
#             print(f"Accuracy till Question no. {i}:========================================= {accuracy:.2f}%")
#             return video_id, json_data
#         if i == len(dict_data) - 1:
#             return None, None
#         else:
#             valid_count += 1
#             if dict_data[video_id]["pred"] == dict_data[video_id]["truth"]:
#                 true_count += 1
#             continue


def unmark_as_processing(file_path, video_id):
    print ("unmark_as_processing")
    dict_data = read_json_file(file_path)

    if video_id in dict_data.keys() and "pred" in dict_data[video_id]:
        del dict_data[video_id]["pred"]
        lock = FileLock(file_path + '.lock')
        with lock, open(file_path, "w") as f:
            json.dump(dict_data, f, indent=4)
        return True
    return False

def save_result(file_path, agent_name:str, video_id:str, agent_prompts:dict, agent_response:dict, prediction_result:int, tool_calls:list, save_backup=False):
    print("SAVING")
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({}, f)
        print(f"Created new result file: {file_path}")


    results = read_json_file(file_path)

    # Ensure keys exist
    if video_id not in results:
        results[video_id] = {}
    if agent_name not in results[video_id]:
        results[video_id][agent_name] = {}

    results[video_id][agent_name]["agent_prompts"] = agent_prompts
    results[video_id][agent_name]["response"] = agent_response
    results[video_id][agent_name]["prediction"] = prediction_result
    results[video_id][agent_name]["tool_calls"] = tool_calls

    # save result
    lock = FileLock(file_path + '.lock')
    with lock, open(file_path, "w") as f:
        json.dump(results, f, indent=4)
        #print(f"Saving this data: {results[video_id][agent_name]} to file_path for video: {video_id}")
    #print(f"Saved results for video_id={video_id}, agent_name={agent_name}")

    # Backup
    from datetime import datetime
    if save_backup == True:
        current_time = datetime.now()
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S') + ".json"
        lock = FileLock(file_path + '.lock')
        with lock, open("backup_" + time_str, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Backup saved: backup_{time_str}")


def get_video_summary(summary_cache_json_path:str, vid:str):
    # Check if the JSON file exists
    if not summary_cache_json_path or not os.path.exists(summary_cache_json_path):
        print("file not found {}".format(summary_cache_json_path))
        return ""

    # Load the JSON file
    with open(summary_cache_json_path, "r", encoding="utf-8") as f:
        video_summaries = json.load(f)

    # Check if the vid result is already in the JSON
    if vid in video_summaries:
        return video_summaries[vid]
    else:
        print(f"Summary for vid '{vid}' not found in JSON.")
        return ""


def create_summary_of_video(openai_api_key="", temperature=0.0, image_dir="", vid="", sampling_interval_sec=3, segment_frames_num=90):
    
    print("create_summary_of_video function called")
    
    # JSON file path
    json_path = os.environ["SUMMARY_CACHE_JSON_PATH"]
    
    # JSON file handling
    if os.path.exists(json_path):
        lock = FileLock(json_path + '.lock')
        with lock, open(json_path, "r") as f:
            video_summaries = json.load(f)
    else:
        video_summaries = {}

    # Check if the vid result is already in the JSON
    if vid in video_summaries:
        print(f"Summary for vid '{vid}' found in JSON. Returning cached result.")
        return video_summaries[vid]
    
    # OpenAI settings
    model_name = "gpt-4o"
    detail     = "low" # high
    client     = OpenAI(api_key=openai_api_key)
    
    # Pricing per 1000 tokens (adjust according to actual pricing)
    model_pricing = {
        "gpt-4o": {
            "prompt": 2.5 / 1000000,      # $2.5 per 1M prompt tokens
            "completion": 10.0 / 1000000  # $10 per 1M completion tokens
        }
    }
    
    # system prompt
    system_prompt = """
    Create a summary of the video based on the sequence of input images. 

    # Output Format

    Provide the summary as a concise paragraph, emphasizing key events or topics represented in the image sequence.
    """
    
    system_prompt_entire = """
    Create a summary of the video based on the provided list of text summaries for each specified segment of the video.

    # Output Format

    Provide the summary as a concise paragraph, emphasizing key events or topics covered throughout the entire video.
    """

    # Get the list of image files
    frame_path_list = sorted(glob.glob(os.path.join(image_dir, vid, "*")))
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    frame_path_list = [path for path in frame_path_list if os.path.splitext(path)[1].lower() in valid_extensions]
    sampled_frame_path_list = frame_path_list[::sampling_interval_sec]
    
    summary_results = []
    temp_frames     = []
    total_tokens    = 0
    total_cost      = 0.0
    
    # create summary every segment_frames_num frames
    for i in range(len(sampled_frame_path_list)):
        data_url = local_image_to_data_url(sampled_frame_path_list[i])
        temp_frames.append({ "type": "image_url", "image_url": { "url": data_url, "detail": detail } })
        
        if len(temp_frames) == segment_frames_num or i == len(sampled_frame_path_list) - 1:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    { "role": "system", "content": system_prompt },
                    { "role": "user", "content": temp_frames }
                ],
                max_tokens=3000,
                temperature=temperature
            )
            summary_results.append(response.choices[0].message.content)
            
            # Get usage data
            # print ("usage: ", response.usage)
            total_tokens += response.usage.total_tokens
            total_cost += (response.usage.prompt_tokens * model_pricing[model_name]['prompt'] + response.usage.completion_tokens * model_pricing[model_name]['completion'])
            
            temp_frames = []
        
    # create entire summary from the summary results
    detail_summaries = ""
    for i in range(len(summary_results)):
        start_sec = i * sampling_interval_sec * segment_frames_num
        end_sec = start_sec + sampling_interval_sec * segment_frames_num
        detail_summaries += f"--- Segment {start_sec}-{end_sec} sec ---\n {summary_results[i]}\n\n"
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            { "role": "system", "content": system_prompt_entire },
            { "role": "user", "content": detail_summaries }
        ],
        max_tokens=10000,
        temperature=temperature
    )
    entire_summary = response.choices[0].message.content
    
    # Get usage data for the entire summary
    # print ("usage: ", response.usage)
    total_tokens += response.usage.total_tokens
    total_cost += (response.usage.prompt_tokens * model_pricing[model_name]['prompt'] + response.usage.completion_tokens * model_pricing[model_name]['completion'])
    
    # Save the result to the JSON
    video_summaries[vid] = {
        "entire_summary": entire_summary,
        "detail_summaries": detail_summaries,
        "total_tokens": total_tokens,
        "total_cost": total_cost
    }
    
    lock = FileLock(json_path + '.lock')
    with lock, open(json_path, "w") as file:
        json.dump(video_summaries, file, indent=4)
    
    return video_summaries[vid]


def prepare_intermediate_steps(steps):

    sanitized = []
    for step in steps:
        action, output = step
        if hasattr(action, "__dict__"):
            action_dict = action.__dict__
        elif isinstance(action, dict):
            action_dict = action
        else:
            action_dict = action

        filtered_action = {}
        if "tool" in action_dict:
            filtered_action["tool"] = action_dict["tool"]
        if "tool_input" in action_dict:
            filtered_action["tool_input"] = action_dict["tool_input"]

        sanitized.append((filtered_action, output))
    return sanitized



if __name__ == "__main__":
    
    # set environment variables
    os.environ["SUMMARY_CACHE_JSON_PATH"] = "/root/VideoMultiAgents/summary_cache_egoschema.json"
    res = create_summary_of_video(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0, image_dir="/root/nas_Ego4D/egoschema/images", vid="55c9c777-a4a9-48df-b0e3-7ebf55788373", sampling_interval_sec=3, segment_frames_num=90)
    print ("*******")
    print(json.dumps(res, indent=2, ensure_ascii=False))
