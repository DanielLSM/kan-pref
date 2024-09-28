import os
from concurrent.futures import ThreadPoolExecutor


# eval "$(micromamba shell hook --shell=bash)"
# micromamba activate
# Define the base command and the different arguments
base_command = (
    "eval \"$(micromamba shell hook --shell=bash)\" && micromamba activate "
    "&& cd /home/mambauser/code/rl_zoo3/ && python3 train_PEBBLE.py "
    "agent.params.actor_lr=0.0005 "
    "agent.params.critic_lr=0.0005 "
    "gradient_update=1 "
    "activation=tanh "
    "num_unsup_steps=9000 "
    "num_train_steps=500000 "
    "num_interact=10000 "
    "max_feedback=500 "
    "reward_batch=25 "
    "reward_update=50 "
    "feed_type= "
    "teacher_beta=-1 "
    "teacher_gamma=1 "
    "teacher_eps_mistake=0.1 "
    "teacher_eps_skip=0 "
    "teacher_eps_equal=0 "
    "agent.params.batch_size=512 "
    "double_q_critic.params.hidden_dim=256 "
    "double_q_critic.params.hidden_depth=3 "
    "diag_gaussian_actor.params.hidden_dim=256 "
    "diag_gaussian_actor.params.hidden_depth=3"
)

docker_image = "dlsm666/kan-pref-cpu:latest"

different_args = {
    'seed': [1,2],  # Example arguments to vary
    'environment': ['metaworld_drawer-close-v2'],  # Different environments
    'reward_model': ['KAN'],  # Changed reward models
    'width': [4, 4, 4],
    'k': 3,
    'grid': 3
}

commands = []
for seed in different_args['seed']:
    for env in different_args['environment']:
        for reward in different_args['reward_model']:
            command = f"{base_command} " + " ".join([  # Changed to use a list
                f"env={env}",
                f"reward_model={reward}",
                f"seed={seed}",
                f"width={[','.join(map(str, different_args['width']))]}",
                f"k={different_args['k']}",
                f"grid={different_args['grid']}",
                f"hydra.run.dir=./docker_logs/{env}/{reward}/{seed}"
            ])  # Modify the command with the arguments
            commands.append(command)

def run_docker(command):
    os.system(f"docker run --user root --rm {docker_image} bash -c '{command}'")  # Added bash -c

def main():
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(run_docker, command) for command in commands]
        for future in futures:
            future.result()  # Wait for all commands to complete

if __name__ == '__main__':
    main()
