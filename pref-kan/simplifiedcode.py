import torch

def prepare_training_data(self):
        #Prepares data for classification
        o1, o2, prefs, critical_points = self.hc.get_all_preference_pairs_with_critical_points() # You can just parse the pickle you loaded directly here
        tensor_o1 = torch.Tensor(o1)
        tensor_o2 = torch.Tensor(o2)
        tensor_prefs = torch.Tensor(prefs)
        tensor_critical_points = torch.Tensor(critical_points)
        my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs, tensor_critical_points)
        my_dataloader = DataLoader(my_dataset, batch_size=self.hc.batch_size, shuffle=True)
        return my_dataloader

def get_all_preference_pairs_with_critical_points(self):
        pairs = [self.pairs[idx] for idx in range(self.pairs_size)]
        obs1, obs2, prefs, critical_points = self.generate_data_for_training_with_critical_points(pairs)
        return obs1, obs2, prefs, critical_points


def train_dataset_with_critical_points(self, dataset, meta_data, epochs_override=-1):
        epochs = epochs_override if epochs_override != -1 else self.training_epochs
        self.reward_model.train(True)
        for epoch in range(1, epochs + 1):

            for step, (o1, o2, prefs, critical_points) in enumerate(dataset):
                self.optimizer.zero_grad()
                o1_unrolled = torch.reshape(o1, [-1, self.obs_size[0] + self.action_size])
                o2_unrolled = torch.reshape(o2, [-1, self.obs_size[0] + self.action_size])
                r1_unrolled = self.reward_model(o1_unrolled)
                r2_unrolled = self.reward_model(o2_unrolled)

                r1_rolled = torch.reshape(r1_unrolled, o1.shape[0:2])
                r2_rolled = torch.reshape(r2_unrolled, o2.shape[0:2])

                rs1 = torch.sum(r1_rolled, dim=1)
                rs2 = torch.sum(r2_rolled, dim=1)
                rss = torch.stack([rs1, rs2])
                rss = torch.t(rss)

                preds = torch.softmax(rss, dim=0)
                preds_correct = torch.eq(torch.argmax(prefs, 1), torch.argmax(preds, 1)).type(torch.float32)
                accuracy = torch.mean(preds_correct)

        return meta_data


my_dataloader = self.prepare_training_data()
self.hc.train_dataset_with_critical_points(my_dataloader, None, training_epochs)