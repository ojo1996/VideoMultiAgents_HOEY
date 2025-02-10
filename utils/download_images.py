from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
import os

# Azure Storage connection string
AZURE_STORAGE_CONNECTION_STRING = "Your Azure Storage Connection String"

# Local directory where the containers' contents will be downloaded (adjust as needed)
DESTINATION_DIRECTORY = "/root/download_directory"

def download_all_containers(destination_directory):
    """
    Download the contents of all containers.
    For each container, create a folder with the same name under the specified directory,
    and save the blobs into that folder.
    """
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    # Get all containers
    containers = blob_service_client.list_containers()

    for container in containers:
        container_name = container['name']
        print(f"\nStarting download for container '{container_name}'...")

        container_client = blob_service_client.get_container_client(container_name)
        # Create a local folder with the same name as the container
        container_folder = os.path.join(destination_directory, container_name)
        os.makedirs(container_folder, exist_ok=True)

        # Get all blobs in the container
        blobs = container_client.list_blobs()
        for blob in blobs:
            blob_name = blob.name
            print(f"  Downloading blob '{blob_name}'...")

            blob_client = container_client.get_blob_client(blob_name)
            # Generate the local file path (recreate subdirectories if present in blob name)
            local_file_path = os.path.join(container_folder, blob_name)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            try:
                with open(local_file_path, "wb") as file:
                    download_stream = blob_client.download_blob()
                    file.write(download_stream.readall())
                print(f"  Saved to {local_file_path}.")
            except Exception as e:
                print(f"  Error: Failed to download blob '{blob_name}': {e}")

    print("\nFinished downloading all containers.")


if __name__ == "__main__":
    download_all_containers(DESTINATION_DIRECTORY)
