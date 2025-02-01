import os
import time
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
from azure.batch.models import (
    TaskState,
    ComputeNodeState,
    PoolResizeParameter
)

###############################################################################################################
###################### ONLY MODIFY THE FOLLOWING VARIABLES TO RUN THE SCRIPT ##################################
###############################################################################################################

BATCH_ACCOUNT_NAME = "your_batch_account_name"
BATCH_ACCOUNT_KEY  = "your_batch_account_key"
BATCH_ACCOUNT_URL  = f"https://{BATCH_ACCOUNT_NAME}.japaneast.batch.azure.com"

###############################################################################################################
###############################################################################################################


def get_pool_job_mapping(batch_client: BatchServiceClient) -> dict:
    """
    Retrieves a list of jobs in the Batch account and returns a mapping
    of pool IDs to a list of associated job IDs.

    Example:
        {
            "poolA": ["job1", "job2"],
            "poolB": ["job3"],
            ...
        }
    """
    jobs = batch_client.job.list()
    pool_job_mapping = {}

    for job in jobs:
        # job.execution_info may be None if the job is not fully set up
        if job.execution_info and job.execution_info.pool_id:
            pool_id = job.execution_info.pool_id
            job_id = job.id
            if pool_id not in pool_job_mapping:
                pool_job_mapping[pool_id] = []
            pool_job_mapping[pool_id].append(job_id)

    return pool_job_mapping


def get_task_status_count(batch_client: BatchServiceClient, job_id: str) -> dict:
    """
    Retrieves tasks for the given job_id and returns a dictionary
    of how many tasks are in each state.

    Example return:
        {
            "active": 10,
            "running": 5,
            "completed": 2
        }
    """
    tasks = batch_client.task.list(job_id)
    status_count = {"active": 0, "running": 0, "completed": 0}

    for task in tasks:
        if task.state == TaskState.active:
            status_count["active"] += 1
        elif task.state == TaskState.running:
            status_count["running"] += 1
        elif task.state == TaskState.completed:
            status_count["completed"] += 1

    return status_count


def get_pool_status_count(batch_client: BatchServiceClient, pool_id: str) -> dict:
    """
    Retrieves a list of compute nodes for the given pool_id and
    returns a dictionary of the count of nodes in each state.

    Example return:
        {
            "idle": 3,
            "running": 5,
            "offlined": 1,
            ...
        }
    """
    nodes = batch_client.compute_node.list(pool_id)
    status_count = {state.value: 0 for state in ComputeNodeState}

    for node in nodes:
        if node.state.value in status_count:
            status_count[node.state.value] += 1

    return status_count


def set_low_priority_nodes(batch_client: BatchServiceClient, pool_id: str, target_low_priority_nodes: int) -> None:
    """
    Sets the number of Spot VMs (low-priority nodes) for the specified pool,
    then triggers a resize operation (does not block until completion here).
    """
    pool = batch_client.pool.get(pool_id)

    resize_params = PoolResizeParameter(
        target_dedicated_nodes=pool.target_dedicated_nodes,
        target_low_priority_nodes=target_low_priority_nodes
    )
    batch_client.pool.resize(pool_id, resize_params)
    print(f"Waiting for pool '{pool_id}' to resize to {target_low_priority_nodes} Spot VMs...")


def delete_unusable_nodes(batch_client: BatchServiceClient, pool_id: str) -> None:
    """
    Removes unusable/abnormal nodes by resizing the pool to only keep healthy nodes.
    This function checks for nodes in specific "unusable" states and resizes
    the pool to exclude those nodes.
    """
    nodes = list(batch_client.compute_node.list(pool_id))

    unusable_states = ["unusable", "starttaskfailed", "offline", "unknown"]
    unusable_nodes = [node for node in nodes if node.state and node.state.lower() in unusable_states]
    healthy_nodes = [node for node in nodes if node.state and node.state.lower() not in unusable_states]

    print(
        f"{pool_id} : Total nodes: {len(nodes)}, Unusable nodes: {len(unusable_nodes)}, "
        f"Healthy nodes: {len(healthy_nodes)}"
    )

    if not unusable_nodes:
        return

    target_low_priority_nodes = len(healthy_nodes)
    print(f"Resizing pool '{pool_id}' to {target_low_priority_nodes} Spot VMs to remove unusable nodes.")
    set_low_priority_nodes(batch_client, pool_id, target_low_priority_nodes)


def get_target_low_priority_nodes(batch_client: BatchServiceClient, pool_id: str) -> int:
    """
    Retrieves the current target_low_priority_nodes setting for the given pool.
    """
    pool = batch_client.pool.get(pool_id)
    return pool.target_low_priority_nodes


def is_pool_resizing(batch_client, pool_id):
    pool = batch_client.pool.get(pool_id)
    if pool.allocation_state is None:
        return False
    return pool.allocation_state.lower() == "resizing"


def manage_pools(batch_client: BatchServiceClient, completed_jobs: set, max_node_count: int = 300):
    """
    Performs node management including scaling up/down and removing unusable nodes.
    Gathers a mapping of pools to jobs, then iterates through each pool:
        0. Skips if pool is currently resizing
        1. Removes unusable nodes
        2. Calculates the total active tasks from ALL jobs in that pool (excluding "completed" jobs)
        3. Scales up if tasks are waiting
        4. Scales down if tasks are finished and there are surplus nodes

    Args:
        batch_client (BatchServiceClient): The Azure Batch client.
        completed_jobs (set): A set of job IDs that have already been identified as completed.
        max_node_count (int): Maximum number of nodes to scale to (default=300).
    """
    print("*******************************************")
    pool_job_mapping = get_pool_job_mapping(batch_client)

    # For each pool, we'll sum active tasks from jobs that are NOT in the "completed_jobs" cache
    for pool_id, job_ids in pool_job_mapping.items():

        # 0. Skip if pool is currently resizing
        if is_pool_resizing(batch_client, pool_id):
            print(f"Pool '{pool_id}' is currently resizing. Skipping this iteration.")
            continue

        # 1. Remove unusable nodes first
        delete_unusable_nodes(batch_client, pool_id)

        # 2. Calculate the total active tasks from all relevant jobs in this pool
        total_active_tasks = 0
        for job_id in job_ids:
            # Skip if job is already flagged as completed
            if job_id in completed_jobs:
                continue

            status_count = get_task_status_count(batch_client, job_id)
            active_count = status_count["active"]
            running_count = status_count["running"]

            # If this job has no active or running tasks, mark it completed
            if active_count == 0 and running_count == 0:
                print(f"Job '{job_id}' appears fully completed. Caching it as completed.")
                completed_jobs.add(job_id)
            else:
                total_active_tasks += active_count

        # If after skipping completed jobs we have no tasks, let's see if we need scale down
        pool_status = get_pool_status_count(batch_client, pool_id)
        unassigned_vm_count = (
            pool_status["idle"] +
            pool_status["creating"] +
            pool_status["starting"] +
            pool_status["waitingforstarttask"] +
            pool_status["rebooting"] +
            pool_status["reimaging"]
        )
        required_vm_count = total_active_tasks - unassigned_vm_count
        current_node_count = get_target_low_priority_nodes(batch_client, pool_id)

        print(f"Pool '{pool_id}' total active tasks (excluding completed jobs): {total_active_tasks}")
        print(f"Pool '{pool_id}' unassigned VMs: {unassigned_vm_count}")
        print(f"Pool '{pool_id}' required VMs to handle tasks: {required_vm_count}")
        print(f"Pool '{pool_id}' current node count: {current_node_count}")

        # 3. Scale up if needed
        if required_vm_count > 0:
            target_vm_count = min(current_node_count + required_vm_count, max_node_count)
            if target_vm_count != current_node_count:
                print(
                    f"Scaling UP pool '{pool_id}': "
                    f"from {current_node_count} to {target_vm_count} low-priority nodes."
                )
                set_low_priority_nodes(batch_client, pool_id, target_vm_count)

        # 4. Scale down if no tasks remain in that pool
        elif total_active_tasks == 0 and unassigned_vm_count > 0:
            target_vm_count = current_node_count - unassigned_vm_count
            target_vm_count = max(target_vm_count, 0)
            if target_vm_count != current_node_count:
                print(
                    f"Scaling DOWN pool '{pool_id}': "
                    f"from {current_node_count} to {target_vm_count} low-priority nodes."
                )
                set_low_priority_nodes(batch_client, pool_id, target_vm_count)


def main():
    """
    Main loop that runs the scaling process every minutes.
    We also maintain a local cache (completed_jobs) of completed job IDs
    so that we do not re-check them until the cache is cleared.

    The cache is cleared every hour to allow for the possibility that a
    previously completed job might have new tasks added in the future.
    """
    credentials = SharedKeyCredentials(BATCH_ACCOUNT_NAME, BATCH_ACCOUNT_KEY)
    batch_client = BatchServiceClient(credentials, batch_url=BATCH_ACCOUNT_URL)

    completed_jobs = set()      # Cache of job IDs that are already completed
    last_cache_clear_time = time.time()
    cache_clear_interval = 3600 # 1 hour in seconds

    while True:
        try:
            now = time.time()
            # Clear the cache if it's been more than an hour
            if now - last_cache_clear_time > cache_clear_interval:
                print("Clearing completed_jobs cache...")
                completed_jobs.clear()
                last_cache_clear_time = now

            manage_pools(batch_client, completed_jobs, max_node_count=300)

        except Exception as e:
            print(f"An error occurred while managing pools: {e}")
        finally:
            time.sleep(60)  # Wait a minutes before the next iteration


if __name__ == "__main__":
    main()
