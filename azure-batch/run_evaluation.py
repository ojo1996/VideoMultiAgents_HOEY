import os
import sys
import json
import time
import azure.batch.models as batchmodels
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
from azure.batch.models import VirtualMachineConfiguration
from azure.batch.models import ImageReference
from azure.batch.models import ContainerConfiguration
from azure.batch.models import PoolAddParameter
from azure.batch.models import ContainerRegistry
from azure.batch.models import PoolInformation
from azure.batch.models import JobAddParameter
from azure.batch.models import JobConstraints


###############################################################################################################
###################### ONLY MODIFY THE FOLLOWING VARIABLES TO RUN THE SCRIPT ##################################
###############################################################################################################
# Azure Batch account settings
BATCH_ACCOUNT_NAME = "YourBatchAccountName"
BATCH_ACCOUNT_URL  = f"https://{BATCH_ACCOUNT_NAME}.japaneast.batch.azure.com"
BATCH_ACCOUNT_KEY  = "YourBatchAccountKey"

# ACR settings
ACR_SERVER   = "YourACRServerURL"
ACR_USERNAME = "YourACRUsername"
ACR_PASSWORD = "YourACRPassword"

# Environment settings
BLOB_CONNECTION_STRING   = "YourBlobConnectionString"
COSMOS_CONNECTION_STRING = "YourCosmosConnectionString"
OPENAI_API_KEY           = "sk-xxxxxx"
EXPERIMENT_ID            = "egoschema_subset"
DATASET                  = "egoschema"
ANNOTATION_FILE          = "/path/to/subset_anno.json"

# Job settings. Do not use latest image tag. Use the specific image tag.
CONTAINER_IMAGE = "acrstanford.azurecr.io/vide-multi-agents:xxxxxx"
POOL_ID         = "pool-video-agents"
JOB_ID          = EXPERIMENT_ID

###############################################################################################################
###############################################################################################################

# Azure Batch init
credentials = SharedKeyCredentials(BATCH_ACCOUNT_NAME, BATCH_ACCOUNT_KEY)
batch_client = BatchServiceClient(credentials, batch_url=BATCH_ACCOUNT_URL)


def read_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        time.sleep(1)
        read_json_file(file_path)


def create_pool( batch_client:BatchServiceClient, pool_id:str, acr_url:str, acr_username:str, acr_password:str):

    # Check if the pool exists
    if pool_id in [pool.id for pool in batch_client.pool.list()]:
        # print(f'Pool {pool_id} already exists')
        pass
    else:
        # Create the pool
        pool = PoolAddParameter(
            id=pool_id,
            vm_size='STANDARD_DS2_V2',
            virtual_machine_configuration=VirtualMachineConfiguration(
                image_reference=ImageReference(publisher="microsoft-dsvm", offer="ubuntu-hpc", sku="2204", version="latest"),
                node_agent_sku_id="batch.node.ubuntu 22.04",
                container_configuration=ContainerConfiguration(
                    type='dockerCompatible',
                    container_registries=[
                        ContainerRegistry(registry_server=acr_url, user_name=acr_username, password=acr_password)
                    ]
                ),
            ),
            enable_auto_scale=False, # Disable auto-scaling because AzureBatchManager will handle it.
            target_dedicated_nodes=0,
            target_low_priority_nodes=0,
            target_node_communication_mode='simplified',
        )
        batch_client.pool.add(pool)

        # Check if the pool is created
        time.sleep(10)
        if pool_id in [pool.id for pool in batch_client.pool.list()]:
            print(f'Pool {pool_id} created successfully')
        else:
            print(f'Failed to create Pool {pool_id}')


def create_job(batch_client: BatchServiceClient, job_id: str, pool_id: str):
    if job_id not in [job.id for job in batch_client.job.list()]:
        job = JobAddParameter(
            id=job_id,
            pool_info=PoolInformation(pool_id=pool_id),
            constraints=JobConstraints(max_task_retry_count=1),
        )
        batch_client.job.add(job)
    else:
        print(f"Job {job_id} already exists. Please rename the EXPERIMENT_ID.")
        sys.exit(1)



if __name__ == '__main__':

    # create pool and job
    create_pool(batch_client, POOL_ID, ACR_SERVER, ACR_USERNAME, ACR_PASSWORD)
    create_job(batch_client, JOB_ID, POOL_ID)

    # prepare tasks
    task_list = []


    dict_data = read_json_file(ANNOTATION_FILE)

    for i, (video_id, json_data) in enumerate(dict_data.items()):

        # delete google_drive_id in json_data
        if "google_drive_id" in json_data:
            del json_data["google_drive_id"]
        json_data_str = json.dumps(json_data)
        # continue

        task_id = f"{EXPERIMENT_ID}_{i:04}"

        # container settings
        container_settings = batchmodels.TaskContainerSettings(
            image_name=CONTAINER_IMAGE,
            registry=batchmodels.ContainerRegistry(
                registry_server=ACR_SERVER,
                user_name=ACR_USERNAME,
                password=ACR_PASSWORD
            ),
            container_run_options="--user root"
        )

        # environment settings
        environment_settings = [
            batchmodels.EnvironmentSetting(name="DATASET",                  value=DATASET),
            batchmodels.EnvironmentSetting(name="BLOB_CONNECTION_STRING",   value=BLOB_CONNECTION_STRING),
            batchmodels.EnvironmentSetting(name="COSMOS_CONNECTION_STRING", value=COSMOS_CONNECTION_STRING),
            batchmodels.EnvironmentSetting(name="VIDEO_FILE_NAME",          value=video_id),
            batchmodels.EnvironmentSetting(name="CONTAINER_NAME",           value=video_id),
            batchmodels.EnvironmentSetting(name="EXPERIMENT_ID",            value=EXPERIMENT_ID),
            batchmodels.EnvironmentSetting(name="QA_JSON_STR",              value=json_data_str),
            batchmodels.EnvironmentSetting(name="OPENAI_API_KEY",           value=OPENAI_API_KEY),
        ]

        # create task
        task = batchmodels.TaskAddParameter(
            id=task_id,
            command_line="/bin/bash -c 'python3 /root/VideoMultiAgents/main.py'",
            container_settings=container_settings,
            environment_settings=environment_settings
        )
        task_list.append(task)

    # add tasks to the job
    try:
        batch_client.task.add_collection(JOB_ID, task_list)
        print(f"Added {len(task_list)} tasks to job {JOB_ID}.")
    except batchmodels.BatchErrorException as err:
        print(f"Error adding tasks: {err}")

