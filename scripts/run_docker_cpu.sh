#!/bin/bash
# Launch an experiment using the docker cpu image
cmd_line="$@"
docker_image=dlsm666/kan-pref-cpu:2.0.2

echo "Executing in the docker (cpu image):"
echo $cmd_line
echo $(pwd)

# docker run --user=root --ipc=host \
#  --mount src=C:/Users/Simon/Documents/Github/kan-pref,target=/home/mambauser/code/rl_zoo3,type=bind sholk/rl-baselines3-zoo-cpu:latest\
#   bash -c "cd /root/code/rl_zoo3/ && $cmd_line"
# sleep 10

#docker run --ipc=host sholk/rl-baselines3-zoo-cpu:latest \
#  bash -c "cd /root/code/rl_zoo3/ && whoami && $cmd_line"
#sleep 10

# docker run -it --user=root --ipc=host \
#   --mount src=/home/daniel/dev/kan-pref/docker_experiments,target=/home/mambauser/code/rl_zoo3,type=bind \
#   dlsm666/rl-baselines3-new-cpu:latest\
#  bash -c "cd /home/mambauser/code/rl_zoo3/ && exec bash"


# docker run -it --user=root --ipc=host \
#   --mount src=/home/daniel/dev/kan-pref/docker_experiments,target=/home/mambauser/code/rl_zoo3,type=bind \
#   dlsm666/rl-baselines3-new-cpu:latest\
#  bash -c "cd /home/mambauser/code/rl_zoo3/ && $cmd_line"
# sleep 10


# THESE ARE THE ONES WE WANT TO USE


docker run -it --user=root --ipc=host \
  $docker_image \
  bash -c "cd /home/mambauser/code/rl_zoo3/ && exec bash"


# docker run -it --user=root --ipc=host \
#   $docker_image \
#   bash -c "cd /home/mambauser/code/rl_zoo3/ && $cmd_line"
