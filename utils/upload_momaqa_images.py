from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
import os

# Azure Storage Connection String
AZURE_STORAGE_CONNECTION_STRING = "Your Azure Storage Connection String"
LOCAL_DIRECTORY = "/root/nas_momaqa/images"

def delete_all_containers():
    """ Delete all containers in Azure Blob Storage """
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    # Get all containers
    containers = blob_service_client.list_containers()

    for container in containers:
        container_name = container['name']
        print(f"Deleting container: {container_name} ...")

        try:
            container_client = blob_service_client.get_container_client(container_name)
            container_client.delete_container()
            print(f"Deleted container: {container_name}")
        except Exception as e:
            print(f"Failed to delete container {container_name}: {e}")

def upload_directory_to_blob(container_name, local_folder):
    """ Upload all image files in a directory to Azure Blob Storage """
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    # create container if not exists
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        print(f"Creating container: {container_name}")
        container_client.create_container()

    # upload all image files in the directory
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
    files = [f for f in os.listdir(local_folder) if f.lower().endswith(image_extensions)]

    for i, file_name in enumerate(files):
        local_file_path = os.path.join(local_folder, file_name)
        blob_path = file_name  # upload the file with the same name

        try:
            blob_client = container_client.get_blob_client(blob_path)
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            # print(f"Uploaded ({i+1}/{len(files)}): {local_file_path} → {blob_path} in container {container_name}")
        except Exception as e:
            print(f"Failed to upload {local_file_path}: {e}")

def main():

    # delete_all_containers()

    # get all folders in the local directory
    folders = [d for d in os.listdir(LOCAL_DIRECTORY) if os.path.isdir(os.path.join(LOCAL_DIRECTORY, d))]

    for i, folder_name in enumerate(folders):
        folder_path = os.path.join(LOCAL_DIRECTORY, folder_name)
        print(f"\nProcessing folder: {folder_name} ({i+1}/{len(folders)})")
        container_name = folder_name.lower().replace("_", "").replace("-", "")
        upload_directory_to_blob(container_name, folder_path)

if __name__ == "__main__":
    main()