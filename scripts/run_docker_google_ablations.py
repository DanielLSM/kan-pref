import os
from google.cloud import batch_v1
from google.cloud.batch_v1 import Job, TaskGroup, TaskSpec, Runnable, AllocationPolicy, ComputeResource
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your parameters
project_id = 'kan-pref'
machine_type = 'custom'
# machine_type = 'e2-standard-4'
# machine_type = 'e2-standard-8'
regions = ['europe-west1', 'europe-west2', 'europe-west3', 'europe-west4']  # Replace with your preferred region
parents = [f'projects/{project_id}/locations/{region}' for region in regions]

# Use the image from Google Container Registry if you pushed it there
# container_image = f'gcr.io/{project_id}/rl-baselines3-new-cpu:2.2.0a1'
# Or use the Docker Hub image directly
container_image = 'docker.io/dlsm666/kan-pref-cpu:2.0.3'

# List of different environments to be used in the experiments
environments = [
    'metaworld_drawer-close-v2',
    'metaworld_button-press-v2',
    'metaworld_button-press-wall-v2',
    'walker_walk',
    'cheetah_run',
    'quadruped_walk'
]

# Define different arguments for commands
different_args = {
    'seeds': [1],  # Example arguments to vary
    'environments': ['metaworld_drawer-close-v2'],  # Different environments
    'reward_models': ['KAN'],  # Changed reward models
    'widths': [[4,4]],  # Different widths
    # 'widths': [[4,4],[8,8],[16,16],[32,32]],  # Different widths
    'k': [3],  # Repeat for k
    'grid': [3]  # Repeat for grid
}
commands = []
for seed in different_args['seeds']:
    for env in different_args['environments']:
        for reward in different_args['reward_models']:
            for width in different_args['widths']:
                for k in different_args['k']:
                    for grid in different_args['grid']:
                        command = [
                            "/bin/bash", "-c", 
                            f"eval \"$(micromamba shell hook --shell=bash)\" && micromamba activate && "
                            f'cd /home/mambauser/code/rl_zoo3/ && python3 train_PEBBLE.py '
                            f'env={env} seed={seed} reward_model={reward} '
                            f'agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 '
                            f'gradient_update=1 activation=tanh num_unsup_steps=9000 '
                            f'num_train_steps=500000 num_interact=10000 max_feedback=500 '
                            f'reward_batch=25 reward_update=50 feed_type=0 '
                            f'teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.1 '
                            f'teacher_eps_skip=0 teacher_eps_equal=0 '
                            f'agent.params.batch_size=512 '
                            f'double_q_critic.params.hidden_dim=256 '
                            f'double_q_critic.params.hidden_depth=3 '
                            f'diag_gaussian_actor.params.hidden_dim=256 '
                            f'diag_gaussian_actor.params.hidden_depth=3 '
                            f'k={k} grid={grid} '
                            f'width={[",".join(map(str, width))]} '
                            f'hydra.run.dir=./docker_logs/{env}/{reward}/{seed}'
                        ]
                        commands.append(command)

# This file is used to keep track of container job indices for Google Cloud Batch jobs.
index_file_path = "container_index.txt"

def create_job(command, job_name, parent, machine_type):
    client = batch_v1.BatchServiceClient()

    # Define the container to run
    container = Runnable.Container(
        image_uri=container_image,
        entrypoint=command[0],
        commands=command[1:],
    )

    # Define the runnable
    runnable = Runnable(
        container=container
    )

    # Define the task specification
    if machine_type == 'custom':
        cpu_milli = 16000  # 16 vCPUs
        memory_mib = 16384  # 16 GB of RAM
    else:
        cpu_milli = 4000 if machine_type == 'e2-standard-4' else 8000  # 4 vCPUs or 8 vCPUs
        memory_mib = 16384 if machine_type == 'e2-standard-4' else 32768  # 16 GB or 32 GB of RAM

    task_spec = TaskSpec(
        runnables=[runnable],
        max_retry_count=0,  # Set retries if needed
        max_run_duration={"seconds": 3600 * 72},  # Max run duration of 72 hours
        compute_resource=ComputeResource(
            cpu_milli=cpu_milli,  # Dynamic CPU allocation
            memory_mib=memory_mib  # Dynamic memory allocation
        )    
    )

    # Define the allocation policy with more vCPUs per task
    allocation_policy = AllocationPolicy(
        instances=[
            AllocationPolicy.InstancePolicyOrTemplate(
                policy=AllocationPolicy.InstancePolicy(
                    machine_type=machine_type  # Use the defined machine type
                )
            )
        ]
    )

    # Define the task group
    task_group = TaskGroup(
        task_spec=task_spec,
        task_count=1  # Number of tasks to run in parallel
    )

    # Define the job
    job = Job(
        task_groups=[task_group],
        allocation_policy=allocation_policy,
        logs_policy=batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        )
    )

    # Submit the job
    response = client.create_job(parent=parent, job=job, job_id=job_name)
    print(f'Job {job_name} created: {response.name}')
    return response.name

def read_index():
    if not os.path.exists(index_file_path):
        with open(index_file_path, 'w') as f:
            f.write('0')
        return 0
    else:
        with open(index_file_path, 'r') as f:
            return int(f.read().strip())

def update_index(new_index):
    with open(index_file_path, 'w') as f:
        f.write(str(new_index))

def main():
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = []
        for i, command in enumerate(commands):
            current_index = read_index()
            update_index(current_index + 1)
            job_name = f'job-{current_index}'
            futures.append(executor.submit(create_job, command, job_name, parents[i % len(parents)], machine_type))

        for future in as_completed(futures):
            try:
                job_name = future.result()
                print(f'Job submitted: {job_name}')
            except Exception as exc:
                print(f'Generated an exception: {exc}')

if __name__ == '__main__':
    main()
