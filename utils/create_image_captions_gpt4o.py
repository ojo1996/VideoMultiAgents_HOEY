import os
import json
import base64
import openai
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ProcessPoolExecutor, as_completed

def send_image_bytes_to_gpt4o(image_bytes, prompt, api_key):
    """
    Sends an image (as bytes) to the GPT-4o API to generate a concise caption.

    Args:
        image_bytes (bytes): The image data.
        prompt (str): The captioning prompt.
        api_key (str): Your OpenAI API key.

    Returns:
        str: The generated caption (or an empty string on error).
    """
    try:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        # create OpenAI client
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in generating concise image captions."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]
                }
            ],
            max_tokens=150
        )
        response_dict = response.to_dict()
        caption = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return caption
    except Exception as e:
        print(f"Error in send_image_bytes_to_gpt4o: {e}")
        return ""

def process_container(container_name, connection_string, openai_api_key, caption_prompt):
    """
    Processes all images in a given container, generating captions for each image.

    Args:
        container_name (str): The name of the container.
        connection_string (str): Azure Blob Storage connection string.
        openai_api_key (str): Your OpenAI API key.
        caption_prompt (str): The captioning prompt.

    Returns:
        tuple: (container_name, list of captions)
    """
    captions = []
    try:
        # create BlobServiceClient and ContainerClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        # sort blobs by name
        blobs = sorted(container_client.list_blobs(), key=lambda b: b.name)
        for blob in blobs:
            if not blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # skip non-image files
            blob_client = container_client.get_blob_client(blob)
            try:
                image_bytes = blob_client.download_blob().readall()
                caption = send_image_bytes_to_gpt4o(image_bytes, caption_prompt, openai_api_key)
                captions.append(caption)
                print(f"[{container_name}] Processed {blob.name}: {caption}")
            except Exception as e:
                print(f"Error processing blob {blob.name} in {container_name}: {e}")
    except Exception as e:
        print(f"Error processing container {container_name}: {e}")
    return container_name, captions

def main():

    CONNECTION_STRING = "Your Azure Blob Storage connection string"
    OPENAI_API_KEY    = "sk-xxxx"
    output_filename   = "captions.json"

    # Prompt for generating image captions
    caption_prompt = "Please provide a concise caption for this image, describing the main subjects briefly."
    caption_prompt = "Please provide a concise caption for this first-person view image, describing the main subjects and their actions"

    # Retrieve the list of containers from the Azure Blob Storage account
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    containers = blob_service_client.list_containers()
    container_names = [container['name'] if isinstance(container, dict) else container.name for container in containers]

    # Limit the number of containers for testing
    container_names = container_names[:10]  # limit to the first 10 containers for testing

    result_data = {}
    processed_count = 0
    save_interval = 10

    # Process each container in parallel using a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_container = {
            executor.submit(process_container, name, CONNECTION_STRING, OPENAI_API_KEY, caption_prompt): name
            for name in container_names
        }
        for future in as_completed(future_to_container):
            container_name, captions = future.result()
            result_data[container_name] = captions
            processed_count += 1
            print(f"Container processed: {container_name} ({processed_count}/{len(container_names)})")
            # save results periodically
            if processed_count % save_interval == 0:
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=4, ensure_ascii=False)
                print(f"Saved results after processing {processed_count} containers.")

    # Save the final results to a JSON file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)
    print(f"Final results written to {output_filename}")

if __name__ == "__main__":
    main()
