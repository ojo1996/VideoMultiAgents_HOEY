import os
import json
from azure.cosmos import CosmosClient

###############################################################################################################
###################### ONLY MODIFY THE FOLLOWING VARIABLES TO RUN THE SCRIPT ##################################
###############################################################################################################

connection_string = "YourCosmosConnectionString"
database_name     = "egoschema"
experiment_id     = "egoschema_fullset"
output_file       = f"{experiment_id}.json" # Optional: Set to None to skip saving the data

###############################################################################################################
###############################################################################################################


def calculate_experiment_accuracy(connection_string: str, database_name: str, experiment_id: str, output_file: str = None) -> float:
    """
    Retrieve experiment data from CosmosDB, calculate accuracy, and optionally save data to a JSON file.

    Parameters:
        connection_string (str): CosmosDB connection string
        database_name (str): Database name
        experiment_id (str): Experiment ID
        output_file (str, optional): Path to save the retrieved data as a JSON file

    Returns:
        float: Accuracy of the experiment (percentage of correct predictions)
    """
    # Connect to CosmosDB
    client = CosmosClient.from_connection_string(connection_string)
    database = client.get_database_client(database_name)
    container = database.get_container_client("experiments")

    # Query data for the specified experiment_id
    query = f"SELECT * FROM experiments e WHERE e.experiment_id = '{experiment_id}'"
    results = list(container.query_items(query=query, enable_cross_partition_query=True))

    if not results:
        print(f"No data found for experiment_id: {experiment_id}")
        return 0.0

    # Remove metadata fields
    filtered_results = [
        {k: v for k, v in item.items() if k not in {"_rid", "_self", "_etag", "_attachments", "_ts"}}
        for item in results
    ]

    # Save results to JSON file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {output_file}")

    # Calculate accuracy
    correct = 0
    total = 0

    for item in filtered_results:
        if "pred" in item and "truth" in item:
            total += 1
            if item["pred"] == item["truth"]:
                correct += 1
        else:
            print(f"Warning: Missing 'pred' or 'truth' in item {item.get('id', 'unknown')}")

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"Accuracy for experiment {experiment_id}: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':

    accuracy = calculate_experiment_accuracy(connection_string, database_name, experiment_id, output_file)