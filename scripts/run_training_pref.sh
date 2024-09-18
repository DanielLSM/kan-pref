#!/bin/bash
python train.py --algo ppo --seed 1 --env button-press-v2  -n 300000 --n_queries 50 --n_init_queries 50 --max_queries 500 --truth 90  --prefcallback "pref.callbacks.UpdateRewardFunctionCriticalPoint" #--track --wandb-project-name PrefLearn --wandb-entity sholk
