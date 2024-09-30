#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm
import wandb
from datetime import datetime

from logger import Logger
from replay_buffer import ReplayBuffer
from kan_reward_model import RewardModel as RewardModelKAN
from reward_model import RewardModel as RewardModelMLP
from collections import deque

import utils
import hydra
class Workspace(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        # Set the number of threads for PyTorch to use all available CPU threads
        torch.set_num_threads(os.cpu_count())

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.pretrained_model = cfg.pretrained_model
        self.use_lora = cfg.use_lora
        self.using_surf = cfg.using_surf
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.model_name = cfg.model_name
        self.env_name = cfg.env

        currTime = datetime.now()
        date_time = currTime.strftime("%H:%M:%S-%d/%m/%Y")

        wandb.login(key="062e6f1457bb47fd3c8c6b4aa043be1dd78e06b3")
        run_name = f"{cfg.env}__{cfg.ablation}__{cfg.reward_model}__{cfg.seed}__{cfg.width}__{cfg.grid}__{cfg.k}__{date_time}"
        config = {"n_queries": cfg.max_feedback,
                  "env": cfg.env,
                  "param_k": cfg.k,
                  "param_grid": cfg.grid,
                  "param_width": cfg.width}

        run = wandb.init(
            name=run_name,
            project=f"preflearn-{cfg.env}-{cfg.ablation}",  # Updated to include env before ablation
            entity="dlsmarta",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True  # auto-upload the videos of agents playing the game
        )
        print(cfg.k)
        print(type(cfg.k))
        print(cfg.grid)
        print(type(cfg.grid))
        # instantiating the reward model
        if cfg.reward_model == "KAN":
            self.reward_model = RewardModelKAN(
                self.env.observation_space.shape[0],
                self.env.action_space.shape[0],
                ensemble_size=cfg.ensemble_size,
                size_segment=cfg.segment,
                activation=cfg.activation, 
                lr=cfg.reward_lr,
                mb_size=cfg.reward_batch, 
                large_batch=cfg.large_batch, 
                label_margin=cfg.label_margin, 
                teacher_beta=cfg.teacher_beta, 
                teacher_gamma=cfg.teacher_gamma, 
                teacher_eps_mistake=cfg.teacher_eps_mistake, 
                teacher_eps_skip=cfg.teacher_eps_skip, 
                teacher_eps_equal=cfg.teacher_eps_equal,
                pretrained_model=self.pretrained_model,
                rank=cfg.rank,
                model_name=self.model_name,
                width=cfg.width,
                grid=cfg.grid,
                k=cfg.k,
                model_path=cfg.model_path)
        elif cfg.reward_model == "MLP":
            self.reward_model = RewardModelMLP(
                self.env.observation_space.shape[0],
                self.env.action_space.shape[0],
                ensemble_size=cfg.ensemble_size,
                lr=cfg.reward_lr,
                mb_size=cfg.reward_batch,
                activation=cfg.activation,
                large_batch=cfg.large_batch,
                label_margin=cfg.label_margin,
                teacher_beta=cfg.teacher_beta,
                teacher_gamma=cfg.teacher_gamma,
                teacher_eps_mistake=cfg.teacher_eps_mistake,
                teacher_eps_skip=cfg.teacher_eps_skip,
                teacher_eps_equal=cfg.teacher_eps_equal,
                pretrained_model=self.pretrained_model,
                use_lora=cfg.use_lora,
                rank=cfg.rank,
                model_name=self.model_name,
                model_path=cfg.model_path)
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            #obs, info = self.env.reset()
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                #obs, reward, done, truncate, extra = self.env.step(action)
                obs, reward, done, extra = self.env.step(action)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self):
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if self.cfg.feed_type == 0:
            labeled_queries = self.reward_model.uniform_sampling()
        elif self.cfg.feed_type == 1:
            labeled_queries = self.reward_model.disagreement_sampling()
        elif self.cfg.feed_type == 2:
            labeled_queries = self.reward_model.entropy_sampling()
        elif self.cfg.feed_type == 3:
            labeled_queries = self.reward_model.kcenter_sampling()
        elif self.cfg.feed_type == 4:
            labeled_queries = self.reward_model.kcenter_disagree_sampling()
        elif self.cfg.feed_type == 5:
            labeled_queries = self.reward_model.kcenter_entropy_sampling()
        else:
            raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))
        wandb.log({"metrics/reward_acc": total_acc})

    def run(self):
        if wandb.run is not None:
            wandb.define_metric("metrics/step")
            wandb.define_metric("metrics/*", step_metric="metrics/step")

        episode, episode_reward, done = 0, 0, True
        episode_metric = 0
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        episode_custom_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        interact_count = 0

        first_time_double_training = False  # True to activate co-training
        while self.step < self.cfg.num_train_steps:
            # print("Step: ", self.step)
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/episode_metric', episode_metric, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/custom_reward', episode_custom_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.step == 0:
                    episode_step = 1
                if episode_step == 0:
                    episode_step = 1
                wandb.log({"metrics/episode_metric": episode_metric / episode_step,
                           "metrics/true_episode_reward": true_episode_reward,
                           "metrics/custom_reward": episode_custom_reward,
                           "metrics/step": self.step})
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                    wandb.log({"metrics/episode_success": episode_success,
                               "metrics/true_episode_success": episode_success})

                #obs, info = self.env.reset()
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_metric = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                episode_custom_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward()
                first_time_double_training = False
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            #next_obs, reward, done, truncate, extra = self.env.step(action)
            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.expand_dims(np.concatenate([obs, action], axis=-1), axis=0))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            episode_metric += math.sqrt(action[0]**2 + action[1]**2 + action[2]**2)
            #episode_metric += next_obs[0]
            true_episode_reward += reward

            # Add custom reward here
            custom_reward = self.get_custom_reward(next_obs, action)
            episode_custom_reward += custom_reward

            if not first_time_double_training and False:
                reward += custom_reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            # if self.step % 120000 == 0: was this
            if self.step % 500000 == 0:
                save_dir = os.path.join(self.work_dir, 'saved_models')
                os.makedirs(save_dir, exist_ok=True)
                self.agent.save(save_dir, self.step)
                self.reward_model.save(save_dir, self.step)

                if wandb:
                    print("saving wandb")
                    wandb.save(os.path.join(save_dir, '*.pt'))

            
        save_dir = os.path.join(self.work_dir, 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        self.agent.save(save_dir, self.step)
        self.reward_model.save(save_dir, self.step)
        if wandb:
            print("saving wandb")
            wandb.save(os.path.join(save_dir, '*.pt'))

    def get_custom_reward(self, observation, action=None):
        if self.env_name == "walker_walk":
            return observation[0]
        if self.env_name == "cheetah_run":
            return observation[0]
        if self.env_name == "quadruped_walk":
            return observation[0]
        else: # Else it's metaworld
            return min(observation[0], 0)
            #return -math.sqrt(action[0]**2 + action[1]**2 + action[2]**2) * 10

@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()