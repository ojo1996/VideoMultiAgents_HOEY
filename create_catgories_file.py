import os
import json
import concurrent.futures
from collections import Counter
from openai import OpenAI
from mimetypes import guess_type
import base64
import glob
import numpy as np


RESULT_FILE_PATH = "data/results/egoschema_fullset_anno.json"
OUTPUT_FILE_PATH = "data/results/egoschema_fullset_categories.json"
PERIODIC_SAVE_INTERVAL = 10  # Save results every 10 items

# Load annotation data from input file
with open(RESULT_FILE_PATH, "r") as f:
    anno_data = json.load(f)


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


def select_category(prompt):
    """
    Calls the GPT-4 Omni API to classify the question into categories.
    If an error occurs during conversion, it retries recursively.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    category_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "category",
            "schema": {"type": "object",
                       "properties": {
                           "reasoning": {"type": "string"},
                           "category": {"type": "string", "enum": ["1", "2", "3", "4", "5"]}
                        },
                        "required": ["reasoning", "category"],
                        "additionalProperties": False
                       },
            "strict": True
        }
    }

    res = ask_gpt4_omni(openai_api_key, prompt, temperature=0.0, json_schema=category_schema)
    res = json.loads(res)
    try:
        question_type = int(res["category"])
    except Exception as e:
        print("Error:", res)
        # Retry on error (recursive call)
        return select_category(prompt)
    return question_type

def process_single_item(index, video_id, json_data):
    """
    Processes a single question item:
    - Extracts the question.
    - Fills the classification prompt.
    - Calls the API to obtain the question type.
    - Returns a list with [index, video_id, question, question_type].
    """
    print("-" * 30)
    print(f"Processing {index+1}/{len(anno_data)}")
    question = json_data["question"]

    prompt = """
You are a question type classification system. Your task is to classify the provided question into one of the following categories.

1. Purpose/Goal Identification
    Description: primary goals, intentions, summary, or overarching themes of the video
    Examples:
    Taking into account all the actions performed by c, what can you deduce about the primary objective and focus within the video content?
    What is the overarching theme of the video, considering the activities performed by both characters?

2. Tools and Materials Usage
    Description: how the character engages with specific tools, materials, and equipment
    Examples:
    What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?
    Explain the significance of the peeler and the knife in the video and their respective roles in the preparation process.

3. Key Action / Moment Detection
    Description: identify crucial steps/actions, the influence/rationale of key action/moment/change on the whole task
    Examples:
    Out of all the actions that took place, identify the most significant one related to food preparation and explain its importance in the context of the video.
    Identify the critical steps taken by c to organize and prepare the engine oil for use on the lawn mower, and highlight the importance of these actions in the overall video narrative.

4. Action Sequence Analysis
    Description: compare and contrast different action sequences, relationship between different actions, how characters adjust their approach, efficacy and precision, expertise of the character
    Examples:
    What is the primary sequence of actions performed by c throughout the video, and how do these actions relate to the overall task being performed?
    Considering the sequence of events, what can be inferred about the importance of precision and accuracy in the character's actions, and how is this demonstrated within the video?

5. Character Interaction
    Description: how characters interact and collaborate, how their roles differ
    Examples:
    What was the main purpose of the actions performed by both c and the man throughout the video, and how did their roles differ?
    Describe the general activity in the room and how the different characters and their actions contribute to this environment.

Classify the following question into the most appropriate category:

Question: {question}
"""
    filled_prompt = prompt.format(question=question)
    question_type = select_category(filled_prompt)
    print("Question:", question)
    print("Question type:", question_type)
    # Return the result as a list: [index, video_id, question, question_type]
    return [index, video_id, question, question_type]

def save_results(result_data):
    """
    Saves the current result_data to the output JSON file.
    """
    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(result_data, f, indent=4)
    print(f"Saved {len(result_data)} results to {OUTPUT_FILE_PATH}")

def main(num_workers):
    result_data = []

    # Use ThreadPoolExecutor for parallel processing (assuming API calls are I/O bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, (video_id, json_data) in enumerate(anno_data.items()):
            futures.append(executor.submit(process_single_item, i, video_id, json_data))

        count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                result_data.append(result)
                count += 1
                # Periodically save results after processing a certain number of items
                if count % PERIODIC_SAVE_INTERVAL == 0:
                    save_results(result_data)
            except Exception as e:
                print("Error processing item:", e)

    # Save final results
    save_results(result_data)

    # Calculate and print category statistics
    id_to_name = {
        1: 'Causal/Purpose/Goal Identification',
        2: 'Character Interaction',
        3: 'Descriptive/Object/Location Identification',
        4: 'Key Action/Moment Detection',
        5: 'Temporal/Action Sequence Analysis'
    }
    arr = []
    for data in result_data:
        arr.append(data[3])
    stat = Counter(arr).most_common()
    total_count = len(result_data)

    print("-" * 30)
    print("Category Statistics:")
    for q_type, cnt in stat:
        print(f"{id_to_name[q_type]}: {cnt / total_count * 100:.1f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args.num_workers)
