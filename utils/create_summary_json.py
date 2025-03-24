import os
import json
import glob
import shutil
import base64
import portalocker
import openai
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# Helper function: Convert local image to data URL
# -------------------------------
def local_image_to_data_url(image_path, detail="low"):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    # Here, it's assumed to be PNG (adjust based on file extension as needed)
    return f"data:image/png;base64,{encoded_image}"

# -------------------------------
# Function to generate video summary from a sequence of images (based on reference code)
# -------------------------------
def create_summary_of_video(openai_api_key="", temperature=0.0, image_dir="", vid="",
                            sampling_interval_sec=3, segment_frames_num=90,
                            cache_json="nextqa_video_summaries.json"):
    print(f"[create_summary_of_video] Called for vid: {vid}")

    # Use the passed-in cache file path instead of an environment variable
    json_path = cache_json

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            video_summaries = json.load(f)
    else:
        video_summaries = {}

    if vid.replace("/", "-") in video_summaries:
        print(f"[create_summary_of_video] Summary for vid '{vid}' found in cache.")
        return video_summaries[vid]

    model_name = "gpt-4o"
    detail     = "low"  # or "high"

    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Pricing information (for reference)
    model_pricing = {
        "gpt-4o": {
            "prompt": 2.5 / 1000000,
            "completion": 10.0 / 1000000
        }
    }

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

    # Assume that the list of image files is stored under image_dir/vid
    vid_dir = os.path.join(image_dir, *vid.split("/"))
    frame_path_list = sorted(glob.glob(os.path.join(vid_dir, "*")))
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    frame_path_list = [path for path in frame_path_list if os.path.splitext(path)[1].lower() in valid_extensions]
    sampled_frame_path_list = frame_path_list[::sampling_interval_sec]

    summary_results = []
    temp_frames = []
    total_tokens = 0
    total_cost = 0.0

    # Generate summaries for image segments at fixed intervals
    for i, frame_path in enumerate(sampled_frame_path_list):
        data_url = local_image_to_data_url(frame_path)
        temp_frames.append({ "type": "image_url", "image_url": { "url": data_url, "detail": detail } })

        if len(temp_frames) == segment_frames_num or i == len(sampled_frame_path_list) - 1:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    { "role": "system", "content": system_prompt },
                    { "role": "user", "content": temp_frames }
                ],
                max_tokens=3000,
                temperature=temperature
            )
            summary_text = response.choices[0].message.content
            summary_results.append(summary_text)
            if hasattr(response, "usage"):
                total_tokens += response.usage.total_tokens
                total_cost += (response.usage.prompt_tokens * model_pricing[model_name]['prompt'] +
                               response.usage.completion_tokens * model_pricing[model_name]['completion'])
            temp_frames = []

    # Combine the summaries of each segment to generate an overall summary
    detail_summaries = ""
    for i, summary in enumerate(summary_results):
        start_sec = i * sampling_interval_sec * segment_frames_num
        end_sec = start_sec + sampling_interval_sec * segment_frames_num
        detail_summaries += f"--- Segment {start_sec}-{end_sec} sec ---\n {summary}\n\n"

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            { "role": "system", "content": system_prompt_entire },
            { "role": "user", "content": detail_summaries }
        ],
        max_tokens=10000,
        temperature=temperature
    )
    entire_summary = response.choices[0].message.content
    if hasattr(response, "usage"):
        total_tokens += response.usage.total_tokens
        total_cost += (response.usage.prompt_tokens * model_pricing[model_name]['prompt'] +
                       response.usage.completion_tokens * model_pricing[model_name]['completion'])

    # re-load cache
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            video_summaries = json.load(f)
    else:
        video_summaries = {}

    key_name = vid.replace("/", "-")
    video_summaries[key_name] = {
        "entire_summary": entire_summary,
        "detail_summaries": detail_summaries,
        "total_tokens": total_tokens,
        "total_cost": total_cost
    }

    with open(json_path, "w") as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        json.dump(video_summaries, file, indent=4)

    return video_summaries[key_name]

# -------------------------------
# Function to process videos (image sequences) corresponding to each JSON entry
# -------------------------------
def process_video_summary(entry, azure_connection_string, openai_api_key, temp_root_dir, cache_json,
                            sampling_interval_sec=3, segment_frames_num=90, temperature=0.0):
    """
    1. Generate the Blob container name from entry's "map_vid_vidorid" (e.g., "1104/4882821564" becomes "1104-4882821564")
    2. Download image files from the target container to a temporary local directory
    3. Generate a video summary based on the image sequence
    4. Delete the downloaded images
    """
    try:
        q_uid = entry.get("q_uid")
        map_vid_vidorid = entry.get("map_vid_vidorid")  # e.g., "1104/4882821564"
        if not map_vid_vidorid:
            print(f"[{q_uid}] 'map_vid_vidorid' is missing. Skipping.")
            return q_uid, None

        # Check if the summary is already cached before downloading images
        if os.path.exists(cache_json):
            with open(cache_json, "r") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                video_summaries = json.load(f)
                key_name = map_vid_vidorid.replace("/", "-")
                if key_name in video_summaries:
                    print(f"[{q_uid}] Cached summary for video '{key_name}' found. Skipping download.")
                    return q_uid, video_summaries[key_name]

        container_name = map_vid_vidorid.replace("/", "-")
        print(f"[{q_uid}] Processing video '{map_vid_vidorid}' from container '{container_name}'")

        # Local storage destination: Save under temp_root_dir following the structure of map_vid_vidorid (e.g., ./temp/1104/4882821564)
        local_vid_dir = os.path.join(temp_root_dir, *map_vid_vidorid.split("/"))
        os.makedirs(local_vid_dir, exist_ok=True)

        # Connect to Azure Blob Storage and download image files from the target container
        blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        blobs = container_client.list_blobs()
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
        download_count = 0
        for blob in blobs:
            if not blob.name.lower().endswith(valid_extensions):
                continue
            blob_client = container_client.get_blob_client(blob)
            local_file_path = os.path.join(local_vid_dir, blob.name)
            with open(local_file_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            download_count += 1

        print(f"[{q_uid}] Downloaded {download_count} images to {local_vid_dir}.")

        # Generate a video summary based on the downloaded images
        summary = create_summary_of_video(
            openai_api_key=openai_api_key,
            temperature=temperature,
            image_dir=temp_root_dir,
            vid=map_vid_vidorid,
            sampling_interval_sec=sampling_interval_sec,
            segment_frames_num=segment_frames_num,
            cache_json=cache_json
        )

        print(f"[{q_uid}] Summary generation completed.")

        # Delete the temporary directory (image files)
        shutil.rmtree(local_vid_dir)
        print(f"[{q_uid}] Local images have been deleted.")

        return q_uid, summary
    except Exception as e:
        print(f"[{entry.get('q_uid')}] Error: {e}")
        return entry.get("q_uid"), None

# -------------------------------
# Main process: Read the JSON file and process each entry in parallel
# -------------------------------
def main():
    # --- Settings ---
    QUESTIONS_JSON_PATH     = "nextqa_val_anno.json"  # Path to the JSON file containing question information
    OUTPUT_SUMMARIES_JSON   = "summary_cache_nextqa_val.json"  # Output JSON file name for the results (also used as cache file)
    AZURE_CONNECTION_STRING = "Your_Azure_Storage_Connection_String"
    OPENAI_API_KEY          = "sk-xxxx"
    TEMP_ROOT_DIR           = "./temp"  # Temporary directory for image downloads
    SAMPLING_INTERVAL_SEC   = 1
    SEGMENT_FRAMES_NUM      = 90
    TEMPERATURE             = 0.0

    # Read the JSON file containing question information
    with open(QUESTIONS_JSON_PATH, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    results = {}
    # Process entries in parallel using ThreadPoolExecutor (adjust max_workers as needed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_qid = {}
        for key, entry in questions_data.items():
            future = executor.submit(
                process_video_summary,
                entry,
                AZURE_CONNECTION_STRING,
                OPENAI_API_KEY,
                TEMP_ROOT_DIR,
                OUTPUT_SUMMARIES_JSON,
                SAMPLING_INTERVAL_SEC,
                SEGMENT_FRAMES_NUM,
                TEMPERATURE
            )
            future_to_qid[future] = key

        for future in as_completed(future_to_qid):
            qid, summary = future.result()
            if summary is not None:
                results[qid] = summary
            else:
                results[qid] = {"error": "Processing failed"}

    print(f"Saved all video summary results to {OUTPUT_SUMMARIES_JSON}.")

if __name__ == "__main__":
    main()
