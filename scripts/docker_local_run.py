import os
from concurrent.futures import ThreadPoolExecutor

# Define the base command and the different arguments
base_command = "python3 train_PEBBLE.py agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=10000 max_feedback=500 reward_batch=25 reward_update=50 feed_type= teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.1 teacher_eps_skip=0 teacher_eps_equal=0 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3"

# different_args = {
#     'seed': [1, 2],  # Example arguments to vary
#     'environment': ['metaworld_drawer-close-v2', 'metaworld_button-press-v2'],  # Different environments
#     'model_name': ['drawer_close_reward_model', 'button_press_reward_model'],  # Different model names
#     'reward_model': ['MLP', 'KAN']  # Changed reward models
# }

different_args = {
    'seed': [1, 2],  # Example arguments to vary
    'environment': ['metaworld_drawer-close-v2'],  # Different environments
    'model_name': ['drawer_close_reward_model', 'button_press_reward_model'],  # Different model names
    'reward_model': ['MLP', 'KAN']  # Changed reward models
}

commands = []
for seed in different_args['seed']:
    for env in different_args['environment']:
        for model in different_args['model_name']:
            for reward in different_args['reward_model']:
                command = f"{base_command} env={env} model_name={model} reward_model={reward} --seed={seed}"  # Modify the command with the arguments
                commands.append(command)

def run_docker(command):
    os.system(f"docker run --rm your_docker_image {command}")

def main():
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(run_docker, command) for command in commands]
        for future in futures:
            future.result()  # Wait for all commands to complete

if __name__ == '__main__':
    main()
