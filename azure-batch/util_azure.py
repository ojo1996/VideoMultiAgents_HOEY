import os
from typing import Dict
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient, PartitionKey


def download_blob_data(starage_account_connection_string:str, video_file_name:str, local_download_path:str):
    """
    Download all blobs for the specified video file to the specified local directory.

    Parameters:
    starage_account_connection_string (str): The connection string to the Azure Storage account.
    video_file_name (str): The id of the specified video file.
    local_download_path (str): The path to the local directory where the blobs will be downloaded.
    """

    # Initialize BlobServiceClient
    
    blob_service_client = BlobServiceClient.from_connection_string(starage_account_connection_string)
    container_client = blob_service_client.get_container_client(video_file_name)

    # Create the local download directory if it does not exist
    local_download_path = os.path.join(local_download_path, video_file_name)
    if not os.path.exists(local_download_path):
        os.makedirs(local_download_path)

    # Download all files in the container
    blobs = container_client.list_blobs()

    for blob in blobs:
        blob_name = blob.name
        local_file_path = os.path.join(local_download_path, blob_name)

        # Check if file already exists
        if os.path.exists(local_file_path):
            print(f"Skipping {blob_name}, already exists at {local_file_path}")
            continue

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        print(f"Downloading {blob_name} to {local_file_path} ...")

        # Download the blob and save it locally
        with open(local_file_path, "wb") as file:
            file.write(container_client.download_blob(blob_name).readall())

    print("Download completed!")



def save_experiment_data(connection_string: str, database_name:str, experiment_id: str, data: Dict):
    """
    Save experiment data to CosmosDB (connect to the container and store the data).

    Parameters:
        connection_string (str): CosmosDB connection string
        database_name (str): Database name should be a dataset name
        experiment_id (str): Experiment ID
        data (Dict): The data to be stored (each entry must contain 'pred' and 'truth' keys)

    Raises:
        ValueError: If any entry in the data does not contain 'pred' and 'truth' keys
    """

    if "pred" not in data.keys() or "truth" not in data.keys():
        raise ValueError("Error: 'pred' and 'truth' fields are required in the data.")

    # Connect to CosmosDB
    client = CosmosClient.from_connection_string(connection_string)
    try:
        database = client.get_database_client(database_name)
        database.read()  # Check if the database exists
    except:
        print(f"Database '{database_name}' not found. Creating it...")
        database = client.create_database(database_name)

    try:
        container = database.get_container_client("experiments")
        container.read()  # Check if the container exists
    except:
        print(f"Collection experiments not found. Creating it...")
        container = database.create_container( id="experiments", partition_key=PartitionKey(path="/experiment_id"))

    # Add the experiment ID to the data
    data["experiment_id"] = experiment_id

    # Save data to CosmosDB (updates existing data if found)
    container.upsert_item(data)

    print(f"Experiment {experiment_id} data saved successfully.")