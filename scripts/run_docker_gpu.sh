#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

docker run --gpus all --ipc=host \
--mount src=C:/Users/Simon/Documents/Github/testing/kan-pref,target=/home/mambauser/code/rl_zoo3,type=bind \
  sholk/rl-baselines3-new-cpu:latest \
  bash -c "cd /home/mambauser/code/rl_zoo3/ && $cmd_line"

sleep 10