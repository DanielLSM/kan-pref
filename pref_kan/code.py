import os
import torch
from oracle import HumanCritic

def load_pickle(name):
    with open(name + ".pkl", 'rb') as handle:
        return pkl.load(handle)

env_id = "HalfCheetah-v3"
data_set_name = "/rl_zoopairs_cheetahbuffer"
path = os.getcwd()
print(path)

pairs = load_pickle(path + data_set_name)
traj1 = pairs[0]
traj2 = pairs[1]
prefs = pairs[2]
critical_points = pairs[3]


env = gym.make(env_id)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
oracle = HumanCritic(state_size, action_size)

tensor_o1 = torch.Tensor(traj1)
tensor_o2 = torch.Tensor(traj2)
tensor_prefs = torch.Tensor(prefs)
tensor_critical_points = torch.Tensor(critical_points)
my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs, tensor_critical_points)
my_dataloader = DataLoader(my_dataset, batch_size=self.hc.batch_size, shuffle=True)

oracle.train_dataset_with_critical_points(my_dataloader, my_dataloader_cost, epoch)