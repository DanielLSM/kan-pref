#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line
echo $pwd

docker run --user=root --ipc=host \
 --mount src=C:/Users/Simon/Documents/Github/kan-pref,target=/home/mambauser/code/rl_zoo3,type=bind sholk/rl-baselines3-zoo-cpu:latest\
  bash -c "cd /root/code/rl_zoo3/ && $cmd_line"
sleep 10

#docker run --ipc=host sholk/rl-baselines3-zoo-cpu:latest \
#  bash -c "cd /root/code/rl_zoo3/ && whoami && $cmd_line"
#sleep 10