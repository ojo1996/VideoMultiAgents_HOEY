from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
import os
import concurrent.futures
from typing import List, Tuple, Dict
import tqdm
import json

# Azure Storage connection string
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=blobnextqaimages;AccountKey=Hq4406t+VrJCyb2lBTDVQY306p2S/sSf1UV/a2pPQ9VAYAMi7+ziztblfKajOmbAoDJipyrjnY/I+AStVI27AQ==;EndpointSuffix=core.windows.net"

# Local directory where the containers' contents will be downloaded
DESTINATION_DIRECTORY = "data/nextqa/frames_aligned/"

# Progress file to track downloaded containers
PROGRESS_FILE = "download_progress.json"

# Number of concurrent downloads
MAX_WORKERS = 50

def load_progress() -> Dict[str, Dict]:
    """
    Load the download progress from the progress file.
    
    Returns:
        Dictionary containing progress information for each container
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress: Dict[str, Dict]):
    """
    Save the download progress to the progress file.
    
    Args:
        progress: Dictionary containing progress information for each container
    """
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def is_container_complete(container_name: str, blob_list: List[dict], container_folder: str) -> bool:
    """
    Check if a container is already fully downloaded.
    
    Args:
        container_name: Name of the container
        blob_list: List of blobs in the container
        container_folder: Local folder where container contents are stored
    
    Returns:
        True if all blobs are downloaded, False otherwise
    """
    for blob in blob_list:
        local_file_path = os.path.join(container_folder, blob.name)
        if not os.path.exists(local_file_path):
            return False
        
        # Optional: Check file size matches
        if os.path.getsize(local_file_path) != blob.size:
            return False
    
    return True

def download_blob(args: Tuple[ContainerClient, str, str, str]) -> Tuple[bool, str, str]:
    """
    Download a single blob if it doesn't exist locally.
    
    Args:
        args: Tuple containing (container_client, blob_name, container_folder, container_name)
    
    Returns:
        Tuple of (success: bool, blob_name: str, error_message: str)
    """
    container_client, blob_name, container_folder, container_name = args
    local_file_path = os.path.join(container_folder, blob_name)
    
    # Skip if file already exists and has correct size
    if os.path.exists(local_file_path):
        blob_client = container_client.get_blob_client(blob_name)
        blob_properties = blob_client.get_blob_properties()
        if os.path.getsize(local_file_path) == blob_properties.size:
            return True, blob_name, "already exists"
    
    try:
        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Get blob client and download
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_file_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())
        
        return True, blob_name, ""
    except Exception as e:
        return False, blob_name, str(e)

def download_container(container_name: str, blob_service_client: BlobServiceClient, progress: Dict[str, Dict]) -> Tuple[int, int, List[str]]:
    """
    Download all blobs from a single container in parallel.
    
    Args:
        container_name: Name of the container
        blob_service_client: Azure blob service client
        progress: Dictionary containing download progress information
    
    Returns:
        Tuple of (successful_downloads: int, failed_downloads: int, errors: List[str])
    """
    container_client = blob_service_client.get_container_client(container_name)
    container_folder = os.path.join(DESTINATION_DIRECTORY, container_name)
    os.makedirs(container_folder, exist_ok=True)
    
    # Get list of all blobs
    blobs = list(container_client.list_blobs())
    total_blobs = len(blobs)
    
    # Check if container is already complete
    if is_container_complete(container_name, blobs, container_folder):
        print(f"Container '{container_name}' is already fully downloaded, skipping...")
        progress[container_name] = {"status": "complete", "total": total_blobs, "successful": total_blobs, "failed": 0}
        save_progress(progress)
        return total_blobs, 0, []
    
    # Prepare arguments for parallel download
    download_args = [
        (container_client, blob.name, container_folder, container_name)
        for blob in blobs
    ]
    
    successful_downloads = 0
    failed_downloads = 0
    errors = []
    
    # Create progress bar
    with tqdm.tqdm(total=total_blobs, desc=f"Downloading {container_name}") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all download tasks
            future_to_blob = {
                executor.submit(download_blob, args): args[1]  # args[1] is blob_name
                for args in download_args
            }
            
            # Process completed downloads
            for future in concurrent.futures.as_completed(future_to_blob):
                success, blob_name, error = future.result()
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                    errors.append(f"Failed to download '{blob_name}': {error}")
                pbar.update(1)
    
    # Update progress
    progress[container_name] = {
        "status": "complete" if failed_downloads == 0 else "incomplete",
        "total": total_blobs,
        "successful": successful_downloads,
        "failed": failed_downloads
    }
    save_progress(progress)
    
    return successful_downloads, failed_downloads, errors

def download_all_containers():
    """
    Download the contents of all containers in parallel, with resume capability.
    """
    # Load existing progress
    progress = load_progress()
    
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    
    # Get all containers
    containers = list(blob_service_client.list_containers())
    total_successful = 0
    total_failed = 0
    all_errors = []
    
    print(f"\nFound {len(containers)} containers to process")
    
    # Process each container
    for container in containers:
        container_name = container['name']
        
        # Skip if container is already complete
        if container_name in progress and progress[container_name]["status"] == "complete":
            print(f"\nSkipping completed container '{container_name}'")
            total_successful += progress[container_name]["successful"]
            continue
        
        print(f"\nProcessing container '{container_name}'...")
        
        successful, failed, errors = download_container(container_name, blob_service_client, progress)
        
        total_successful += successful
        total_failed += failed
        all_errors.extend(errors)
        
        print(f"Container '{container_name}' complete: {successful} successful, {failed} failed")
    
    # Print final summary
    print("\nDownload Summary:")
    print(f"Total files successfully downloaded: {total_successful}")
    print(f"Total files failed: {total_failed}")
    
    if all_errors:
        print("\nErrors encountered:")
        for error in all_errors:
            print(f"  {error}")

if __name__ == "__main__":
    download_all_containers()
