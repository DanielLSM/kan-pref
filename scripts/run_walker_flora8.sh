#!/bin/bash
# Activate Micromamba environment
#eval "$(micromamba shell hook --shell=bash)"
#micromamba activate
echo "PATH: $PATH"
export PATH=$PATH:/opt/conda/envs/base/bin
echo "PATH: $PATH"
/opt/conda/bin/python train_PEBBLE.py env=metaworld_drawer-close-v2 seed=3 model_name=drawer_close_reward_model agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=10000 max_feedback=500 reward_batch=25 reward_update=50 feed_type= teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.1 teacher_eps_skip=0 teacher_eps_equal=0 width=[2,2] k=9 grid=3
