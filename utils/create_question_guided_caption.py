import os
import sys
import json
import base64
import argparse
import openai
import spacy
from azure.storage.blob import BlobServiceClient
from filelock import FileLock  # pip install filelock
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def extract_vocab(text):
    """
    Extracts nouns and verbs from the text using spaCy.
    Returns two sets: nouns and verbs.
    """
    doc = nlp(text)
    nouns = {token.text for token in doc if token.pos_ == "NOUN"}
    verbs = {token.text for token in doc if token.pos_ == "VERB"}
    return nouns, verbs

def build_question_vocab(question_item):
    """
    For a given question item (with 'question' and options),
    extracts and aggregates nouns and verbs.
    Returns sorted lists of nouns and verbs.
    """
    all_nouns, all_verbs = set(), set()
    q_text = question_item.get("question", "")
    nouns, verbs = extract_vocab(q_text)
    all_nouns.update(nouns)
    all_verbs.update(verbs)
    for i in range(5):
        opt = question_item.get(f"option {i}", "")
        nouns, verbs = extract_vocab(opt)
        all_nouns.update(nouns)
        all_verbs.update(verbs)
    return sorted(all_nouns), sorted(all_verbs)

def create_chunks(blob_list, chunk_size, overlap):
    """
    Divides a list into chunks of size 'chunk_size' with a specified overlap.
    For example, if chunk_size=5 and overlap=2, the first chunk is [0-4],
    the second chunk is [3-7], etc.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(blob_list), step):
        chunk = blob_list[i:i+chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks

def send_images_to_gpt4o(image_bytes_list, prompt, api_key):
    """
    Sends a list of images (as bytes) along with a text prompt to the GPT-4o API.
    Each image is base64 encoded and included in the payload.
    Returns the generated caption string.
    """
    try:
        images_payload = []
        for image_bytes in image_bytes_list:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            images_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"}
            })
        messages = [
            {
                "role": "system",
                "content": "You are an expert in generating concise image captions that reflect both visual content and temporal sequence."
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + images_payload
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=150
        )
        caption = response["choices"][0]["message"]["content"].strip()
        return caption
    except Exception as e:
        print(f"Error in send_images_to_gpt4o: {e}")
        return ""

def get_image_blobs(container_name, connection_string):
    """
    Connects to the given Azure Blob Storage container and returns a sorted list of blobs
    corresponding to images (files ending with .jpg, .jpeg, or .png).
    """
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    image_extensions = ('.jpg', '.jpeg', '.png')
    blobs = [blob for blob in container_client.list_blobs() if blob.name.lower().endswith(image_extensions)]
    blobs = sorted(blobs, key=lambda b: b.name)
    return blobs

def download_blob_bytes(container_client, blob_name):
    """
    Downloads the blob content as bytes.
    """
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()

def process_question(qid, args):
    """
    Processes a given question id (used as container name) and generates captions.
    Returns a dictionary with container name as key and list of captions as value.
    """
    with open(args.question_json, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    if qid not in questions_data:
        print(f"Question id {qid} not found in JSON file.")
        return {}

    q_item = questions_data[qid]
    print(f"Processing question {qid} in worker.")

    # Extract vocabulary from the question and options
    nouns, verbs = build_question_vocab(q_item)
    vocab_str = f"Nouns: {', '.join(nouns)}; Verbs: {', '.join(verbs)}"

    # Build the prompt including the question, options, and extracted vocabulary
    question_text = q_item.get("question", "")
    options = [q_item.get(f"option {i}", "") for i in range(5)]
    options_str = "; ".join(options)
    prompt = (
        f"Your task is to create short and concise image captions. Describe the image by focusing on the information most relevant to answering the Video Question Answering task questions.\n"
        f"Video Question: {question_text}\n"
        f"Options: {options_str}\n"
        f"Note : Please include key details about the setting (e.g., Key words extracted: {vocab_str}), tools involved, and any noticeable changes. Please DO NOT answer the question directly."
    )

    # Use qid as container name
    # container_name = qid
    container_name = q_item.get("map_vid_vidorid", "").lower().replace("/", "-")
    results = {qid: []}

    try:
        blob_service_client = BlobServiceClient.from_connection_string(args.connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blobs = get_image_blobs(container_name, args.connection_string)
        if not blobs:
            print(f"No image blobs found in container {container_name} for question {qid}.")
            return results
    except Exception as e:
        print(f"Error accessing container {container_name}: {e}")
        return results

    try:
        chunks = create_chunks(blobs, args.chunk_size, args.overlap)
    except Exception as e:
        print(f"Error creating chunks for container {container_name}: {e}")
        return results

    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)} for question {qid}")
        image_bytes_list = []
        for blob in chunk:
            try:
                image_bytes = download_blob_bytes(container_client, blob.name)
                image_bytes_list.append(image_bytes)
            except Exception as e:
                print(f"Error downloading blob {blob.name}: {e}")
        if not image_bytes_list:
            continue
        caption = send_images_to_gpt4o(image_bytes_list, prompt, args.openai_api_key)
        results[qid].append(caption)

    return results

def update_shared_output(output_file, container_name, result_for_container):
    """
    Uses a file lock to update the shared JSON output file.
    It reads the current file, updates the entry for the given container, and writes back.
    """
    lock_path = output_file + ".lock"
    lock = FileLock(lock_path)
    with lock:
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        data[container_name] = result_for_container
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Updated shared output for container {container_name}")

def process_question_worker(qid, args):
    """
    Worker function that processes a single question.
    It updates the shared output file with the results.
    """
    result = process_question(qid, args)
    container_name = qid
    update_shared_output(args.output, container_name, result.get(container_name, []))
    return qid, result

def main():
    parser = argparse.ArgumentParser(
        description="Generate question-specified captions grouped by container name using ProcessPoolExecutor."
    )
    parser.add_argument("--question-json", type=str, required=True, help="Path to JSON file containing questions and options.")
    parser.add_argument("--connection-string", type=str, required=True, help="Azure Blob Storage connection string.")
    parser.add_argument("--openai-api-key", type=str, required=True, help="Your OpenAI API key.")
    parser.add_argument("--chunk-size", type=int, default=5, help="Number of images per chunk.")
    parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping images between consecutive chunks.")
    parser.add_argument("--output", type=str, default="container_captions.json", help="Output JSON filename.")
    parser.add_argument("--max-workers", type=int, default=32, help="Maximum number of worker processes.")

    args = parser.parse_args()
    openai.api_key = args.openai_api_key

    # Read questions data
    with open(args.question_json, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    # Load already processed questions if output file exists (for resuming)
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            processed = json.load(f)
    else:
        processed = {}

    # Prepare list of question IDs that have not been processed yet
    question_ids = [qid for qid in questions_data.keys() if qid not in processed]
    if not question_ids:
        print("All questions have already been processed.")
        return

    # Use ProcessPoolExecutor for parallel processing of questions
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_qid = {executor.submit(process_question_worker, qid, args): qid for qid in question_ids}
        for future in as_completed(future_to_qid):
            qid = future_to_qid[future]
            try:
                result_qid, result = future.result()
                print(f"Worker for question {result_qid} finished processing.")
            except Exception as exc:
                print(f"Question {qid} generated an exception: {exc}")

    print(f"Caption generation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
