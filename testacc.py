import os

import gym
import numpy as np
import torch
import pickle as pkl

from torch.utils.data import TensorDataset, DataLoader

from pref_kan.oracle import HumanCritic
# from pref_kan.kan_oracle import HumanCritic

def load_pickle(name):
    with open(name + ".pkl", 'rb') as handle:
        return pkl.load(handle)

def generate_data_for_training_with_critical_points(queries):
    queries = np.array(queries, dtype=object)
    o1, o2, prefs = queries[:, 0, 0], queries[:, 1, 0], queries[:, 2]
    o1 = [np.stack(segments) for segments in o1]
    o2 = [np.stack(segments) for segments in o2]
    prefs = np.asarray(prefs).astype('float32')
    return o1, o2, prefs

def create_dataloader(pairs):

    traj1, traj2, prefs = generate_data_for_training_with_critical_points(pairs)
    tensor_o1 = torch.Tensor(traj1)
    tensor_o2 = torch.Tensor(traj2)
    tensor_prefs = torch.Tensor(prefs)
    my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs)
    my_dataloader = DataLoader(my_dataset, batch_size=128, shuffle=True)
    return my_dataloader

def test_accuracy(dataset, oracle):
    for step, (o1, o2, prefs) in enumerate(dataset):
            o1 = o1.to('cuda')  # Move input tensors to the device
            o2 = o2.to('cuda')
            prefs = prefs.to('cuda')
            o1_unrolled = torch.reshape(o1, [-1, oracle.obs_size[0] + oracle.action_size])
            o2_unrolled = torch.reshape(o2, [-1, oracle.obs_size[0] + oracle.action_size])
            r1_unrolled = oracle.reward_model(o1_unrolled)
            r2_unrolled = oracle.reward_model(o2_unrolled)

            r1_rolled = torch.reshape(r1_unrolled, o1.shape[0:2])
            r2_rolled = torch.reshape(r2_unrolled, o2.shape[0:2])

            rs1 = torch.sum(r1_rolled, dim=1)
            rs2 = torch.sum(r2_rolled, dim=1)
            rss = torch.stack([rs1, rs2])
            rss = torch.t(rss)

            preds = torch.softmax(rss, dim=0)
            preds_correct = torch.eq(torch.argmax(prefs, 1), torch.argmax(preds, 1)).type(torch.float32)
            accuracy = torch.mean(preds_correct)

            print(f"Accuracy: %s", accuracy.item())



env_id = "HalfCheetah-v3"
data_set_name = "/pref_kan/data/cheetah/rl_zoopairs_cheetahbuffer"
sizes_data_set_name = "/pref_kan/data/cheetah/rl_zoopairs_size_cheetahbuffer"
path = os.getcwd()
print(path)


pairs = load_pickle(path + data_set_name)
pairs_size = load_pickle(path + sizes_data_set_name)

env = gym.make(env_id)
state_size = env.observation_space.shape
action_size = env.action_space.shape[0]
oracle = HumanCritic(state_size, action_size)

training_pairs = [pairs[idx] for idx in range(pairs_size-10)]
training_dataloader = create_dataloader(training_pairs)

oracle.train_dataset(training_dataloader, {})

test_pairs = [pairs[idx] for idx in range(pairs_size-10, pairs_size)]
test_dataloader = create_dataloader(test_pairs)
test_agent = test_accuracy(test_dataloader, oracle)