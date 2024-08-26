import os
import time
import requests
import datetime
from retry import retry
from openai import AzureOpenAI, OpenAI
import re
import json
import random
import glob
import base64
import portalocker
from mimetypes import guess_type


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
def ask_gpt4_omni(openai_api_key="", prompt_text="", temperature=0.0, image_dir="", vid="", frame_num=18, detail="low", use_selected_images=None):
    model_name = "gpt-4o"
    # model_name = "gpt-4o-mini"

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
            step = len(frame_path_list) // frame_num
            start = random.randint(0, int(len(frame_path_list) / frame_num))
            for i in range(start, len(frame_path_list), step):
                data_url = local_image_to_data_url(frame_path_list[i])
                frames.append({ "type": "image_url", "image_url": { "url": data_url, "detail": detail } })

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                { "role": "system", "content": "You are a helpful expert in first person view video analysis." },
                { "role": "user", "content": prompt_text },
                { "role": "user", "content": frames }
            ],
            max_tokens=3000,
            temperature=temperature
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                { "role": "system", "content": "You are a helpful expert in first person view video analysis." },
                { "role": "user", "content": prompt_text }
            ],
            max_tokens=3000,
            temperature=temperature
        )

    return response.choices[0].message.content


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

    prompt = (
        "[Question and 5 Options to Solve]\n"
        f"{question}\n"
        f"{options_str}\n\n"
        "[Instructions]\n"
        "Please identify two experts to answer questions related to this video. Name the two types of experts and specify their fields of expertise.\n"
        "Ensure the expert types come from different fields to provide diverse perspectives.\n"
        "Additionally, create a prompt for each expert to answer the questions. Instruct each expert to provide two answers and explanations.\n\n"
        "[Example prompt for ExpertNameXPrompt]\n"
        "You are a Housekeeping Expert. Watch the video from the perspective of a professional housekeeper and answer the following questions based on your expertise.\n"
        "Please think step-by-step.\n\n"
        "[Response Format]\n"
        "You must respond using this JSON format:\n"
        "{\n"
        '  "ExpertName1": "xxxx",\n'
        '  "ExpertName1Prompt": "xxxx",\n'
        '  "ExpertName2": "xxxx",\n'
        '  "ExpertName2Prompt": "xxxx"\n'
        "}"
    )
    return prompt


def create_question_sentence(question_data:dict, shuffle_questions=False):
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


def create_stage2_agent_prompt(question_data:dict, generated_expert_prompt="", shuffle_questions=False):
    prompt = create_question_sentence(question_data, shuffle_questions)
    prompt += "\n\n[Instructions]\n"
    prompt += "Understand the question and options well and focus on the differences between the options.\n"
    # prompt += "Exclude options that contain unnecessary embellishments, such as subjective adverbs or clauses that cannot be objectively determined, and consider only the remaining options.\n"
    prompt += generated_expert_prompt
    return prompt


def create_stage2_organizer_prompt(question_data:dict, shuffle_questions=False):

    organizer_prompt = (
        "[Instructions]\n"
        "You are the organizer of a discussion. Your task is to analyze the opinions of other Agents and make a final decision.\n"
        "Your output should be one of the following options: OptionA, OptionB, OptionC, OptionD, OptionE, along with an explanation.\n"
        "The correct answer is always within these 5 options and is a simple and straightforward choice.\n"
        "Provide a step-by-step explanation of your reasoning.\n"
        "You should respect the opinions of other experts. Also, include the opinions of other experts in your explanation.\n\n"

        # "Exclude options that contain unnecessary embellishments, such as subjective adverbs or clauses that cannot be objectively determined, and consider only the remaining options.\n"
        # "Place importance on clear and concise expression, and avoid choosing options that include unnecessary embellishments\n"
        # "Avoid choosing options that include adverbs and other unnecessary embellishments (especially those indicating properties or states), and place importance on comprehensive and accurate descriptions of objects and actions in clear sentences.\n\n"
        "Avoid choosing options that include adverbs and other unnecessary embellishments, especially those indicating properties or states\n"
        "Place importance on comprehensive and accurate descriptions of objects and actions in sentences.\n\n"

        f"{create_question_sentence(question_data, shuffle_questions=False)}\n\n"

        "[Output Format]\n"
        "Your response should be formatted as follows:\n"
        "Pred: OptionX\n"
        "Explanation: Your detailed explanation here.\n\n"
    )

    return organizer_prompt


def post_process(response):
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


def extract_expert_info_json(data):
    result = {}

    json_start = data.find('{')
    json_end = data.rfind('}') + 1

    json_data = data[json_start:json_end]

    if json_start != -1:
        try:
            json_extract = json.loads(json_data)
            for key, value in json_extract.items():
                if "ExpertName" in key and "Prompt" not in key:
                    number = re.findall(r'\d+', key)[0]
                    expert_key = f"ExpertName{number}"
                    prompt_key = f"ExpertName{number}Prompt"
                    expert_name = value
                    prompt_text = json_extract.get(prompt_key, "")

                    result[expert_key] = expert_name.strip().replace('"', "'")
                    result[prompt_key] = prompt_text.strip().replace('"', "'")
        except json.JSONDecodeError:
            print("JSONDecodeError: Failed to extract expert information from the response.")

    return result


def extract_expert_info(data):
    result = extract_expert_info_json(data)
    result = add_text_analysis_expert_info(result)
    if len([k for k in result if "ExpertName" in k]) >= 2 and len([k for k in result if "Prompt" in k]) >= 3:
        return result
    else:
        return None


def add_text_analysis_expert_info(data):
    data["ExpertName3"] = "Text Analysis Expert"
    data["ExpertName3Prompt"] = "You are a Text Analysis Expert. For each option, check that the following two points are satisfied and insist on excluding any unsuitable ones.\n 1. The sentence does not contain unnecessary embellishments, for example, subjective adverbs or situational situational statements.\n 2. The sentence is comprehensive and accurate with regard to objects and actions.\n"
    return data

def read_json_file(file_path):
    # print ("read_json_file")
    try:
        with open(file_path, "r") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            data = json.load(f)
            portalocker.unlock(f)
            return data
    except Exception as e:
        time.sleep(1)
        read_json_file(file_path)


def select_data_and_mark_as_processing(file_path):
    print ("select_data_and_mark_as_processing")
    dict_data = read_json_file(file_path)

    for i, (video_id, json_data) in enumerate(dict_data.items()):

        if "pred" not in json_data.keys():
            dict_data[video_id]["pred"] = -2
            with open(file_path, "w") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                json.dump(dict_data, f, indent=4)
                portalocker.unlock(f)
            return video_id, json_data
    return None, None


def unmark_as_processing(file_path, video_id):
    print ("unmark_as_processing")
    dict_data = read_json_file(file_path)

    if video_id in dict_data.keys() and "pred" in dict_data[video_id]:
        del dict_data[video_id]["pred"]
        with open(file_path, "w") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(dict_data, f, indent=4)
            portalocker.unlock(f)
        return True
    return False


def save_result(file_path, video_id:str, expert_info:dict, agent_prompts:dict, agent_response:dict, prediction_result:int, save_backup=False):
    questions = read_json_file(file_path)

    questions[video_id]["expert_info"] = expert_info
    questions[video_id]["agent_prompts"] = agent_prompts
    questions[video_id]["response"] = agent_response
    questions[video_id]["pred"] = prediction_result
    # if result == -1:
    #     # use random value 0 to 4
    #     questions[video_id]["pred"] = random.randint(0, 4)
    #     questions[video_id]["invalid"] = "true"
    # else:
    #     questions[video_id]["pred"] = result

    # save result
    with open(file_path, "w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(questions, f, indent=4)
        portalocker.unlock(f)

    # Backup
    from datetime import datetime
    if save_backup == True:
        current_time = datetime.now()
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S') + ".json"
        with open("backup_" + time_str, "w") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(questions, f, indent=4)
            portalocker.unlock(f)


def save_re_write_question_and_options(file_path, video_id:str, rewrited_qa:dict):
    questions = read_json_file(file_path)

    questions[video_id]["rewrited_qa"] = rewrited_qa

    # save result
    with open(file_path, "w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(questions, f, indent=4)
        portalocker.unlock(f)
