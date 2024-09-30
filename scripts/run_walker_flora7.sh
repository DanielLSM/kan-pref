#python train_PEBBLE.py env=metaworld_button-press-v2 seed=1 model_name=button_press_reward_model agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=400 reward_batch=40 reward_update=50 feed_type= teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.1 teacher_eps_skip=0 teacher_eps_equal=0 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3
#python train_PEBBLE.py env=walker_walk seed=1 model_name=walker_reward_model agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=500 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=5 reward_update=50 feed_type= teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.1 teacher_eps_skip=0 teacher_eps_equal=0 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3
# python train_PEBBLE.py env=metaworld_drawer-close-v2 seed=1 model_name=drawer_close_reward_model agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=10000 max_feedback=500 reward_batch=25 reward_update=50 feed_type= teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.1 teacher_eps_skip=0 teacher_eps_equal=0 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3
python train_PEBBLE_clean.py \
    env=metaworld_drawer-close-v2 \
    seed=1 \
    model_name=drawer_close_reward_model \
    agent.params.actor_lr=0.0005 \
    agent.params.critic_lr=0.0005 \
    gradient_update=1 \
    activation=tanh \
    num_unsup_steps=9000 \
    num_train_steps=500000 \
    num_interact=10000 \
    max_feedback=500 \
    reward_batch=25 \
    reward_update=50 \
    feed_type=0 \
    teacher_beta=-1 \
    teacher_gamma=1 \
    teacher_eps_mistake=0.1 \
    teacher_eps_skip=0 \
    teacher_eps_equal=0 \
    agent.params.batch_size=512 \
    double_q_critic.params.hidden_dim=256 \
    double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 \
    diag_gaussian_actor.params.hidden_depth=3
