import torch

def train_reward_model(self, training_epochs):
    #Probably load the data here somewhere
    self.hc.generate_preference_pairs_uniform(trajectories, critical_points, truth=self.truth, number_of_queries=self.n_initial_queries)
    my_dataloader = self.prepare_training_data()
    self.hc.train_dataset_with_critical_points(my_dataloader, None, training_epochs)


def prepare_training_data(self):
    o1, o2, prefs, critical_points = self.hc.get_all_preference_pairs_with_critical_points()
    tensor_o1 = torch.Tensor(o1)
    tensor_o2 = torch.Tensor(o2)
    tensor_prefs = torch.Tensor(prefs)
    tensor_critical_points = torch.Tensor(critical_points)
    my_dataset = TensorDataset(tensor_o1, tensor_o2, tensor_prefs, tensor_critical_points)
    my_dataloader = DataLoader(my_dataset, batch_size=self.hc.batch_size, shuffle=True)
    return my_dataloader

self.hc.train_dataset_with_critical_points(my_dataloader, None, training_epochs)


def get_all_preference_pairs_with_critical_points(self):
    pairs = [self.pairs[idx] for idx in range(self.pairs_size)]
    obs1, obs2, prefs, critical_points = self.generate_data_for_training_with_critical_points(pairs)
    return obs1, obs2, prefs, critical_points

def save_buffers(self, path="", env_name="", save_name="buffer"):
    save_pickle(self.segments, path + "segments_" + env_name + save_name)
    save_pickle(self.segments_size, path + "segments_size_" + env_name + save_name)
    save_pickle(self.segments_index, path + "segments_index_" + env_name + save_name)
    save_pickle(self.pairs, path + "pairs_" + env_name + save_name)
    save_pickle(self.pairs_size, path + "pairs_size_" + env_name + save_name)
    save_pickle(self.pairs_index, path + "pairs_index_" + env_name + save_name)
    save_pickle(self.critical_points, path + "critical_points_" + env_name + save_name)
    save_pickle(self.critical_points_size, path + "critical_points_size_" + env_name + save_name)
    save_pickle(self.critical_points_index, path + "critical_points_index_" + env_name + save_name)

def load_buffers(self, path="", env_name="", load_name="buffer"):
    self.segments = load_pickle(path + "segments_" + env_name + load_name)
    self.segments_size = load_pickle(path + "segments_size_" + env_name + load_name)
    self.segments_index = load_pickle(path + "segments_index_" + env_name + load_name)
    self.pairs = load_pickle(path + "pairs_" + env_name + load_name)
    self.pairs_size = load_pickle(path + "pairs_size_" + env_name + load_name)
    self.pairs_index = load_pickle(path + "pairs_index_" + env_name + load_name)
    self.pairs = load_pickle(path + "critical_points_" + env_name + load_name)
    self.pairs_size = load_pickle(path + "critical_points_size_" + env_name + load_name)
    self.pairs_index = load_pickle(path + "critical_points_index_" + env_name + load_name)


def train_dataset_with_critical_points(self, dataset, meta_data, epochs_override=-1):

    epochs = epochs_override if epochs_override != -1 else self.training_epochs

    self.reward_model.train(True)
    avg_loss = 0
    meta_data = {}
    episode_loss = 0

    for epoch in range(1, epochs + 1):

        running_loss = 0
        running_accuracy = 0

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

            loss_pref = -torch.sum(torch.log(preds[prefs == 1]))
            loss = loss_pref

            running_loss += loss.detach().numpy().item()
            running_accuracy += accuracy


            reporting_interval = (self.training_epochs // 10) if self.training_epochs >= 10 else 1
            if epoch % reporting_interval == 0 and step == len(dataset) - 1:
                print("Epoch %d , Training loss (for one batch) at step %d: %.4f, Accuracy %.4f" % (epoch, step, float(loss), float(accuracy)))
                print("Seen so far: %s samples" % ((step + 1) * self.batch_size))

            loss.backward()
            self.optimizer.step()

        episode_loss = (running_loss / len(dataset))
        avg_loss += episode_loss
        episode_accuracy = (running_accuracy / len(dataset))
        if self.writer:
            self.writer.add_scalar("reward/loss", episode_loss, self.updates)
            self.writer.add_scalar("reward/accuracy", episode_accuracy, self.updates)
        if wandb.run is not None:
            wandb.log({"reward/loss": episode_loss,
                        "reward/accuracy": episode_accuracy,
                        "reward/updates": self.updates
                        })
        self.updates += 1

    avg_loss = avg_loss / epochs
    if (avg_loss - episode_loss) < 2.5:
        print("episode_loss:" + str(episode_loss))
        print("avg_loss:" + str(avg_loss))
        meta_data['improved'] = False
    else:
        meta_data['improved'] = True
    self.reward_model.train(False)
    return meta_data


def get_query_results_reward(self, segment1, segment2, truth, critical_points):
    total_reward_1 = segment1[-1]
    total_reward_2 = segment2[-1]
    truth_percentage = truth / 100.0
    fakes_percentage = 1 - truth_percentage
    epsilon = self.epsilon
    if total_reward_1 > total_reward_2 + epsilon:
        preference = [1, 0] if fakes_percentage < random.random() else [0, 1]
        point = critical_points[0] if preference[0] == 1 else critical_points[1]

    elif total_reward_1 + epsilon < total_reward_2:
        preference = [0, 1] if fakes_percentage < random.random() else [1, 0]
        point = critical_points[1] if preference[1] == 1 else critical_points[0]
    else:
        preference = [0, 0]
        point = [-1, -1]
    return [segment1, segment2, preference, point]

def generate_data_for_training_with_critical_points(self, queries):
    queries = np.array(queries, dtype=object)
    o1, o2, prefs, critical_points = queries[:, 0, 0], queries[:, 1, 0], queries[:, 2], queries[:, 3]
    o1 = [np.stack(segments) for segments in o1]
    o2 = [np.stack(segments) for segments in o2]

    critical_points = self.generate_critical_point_segment(critical_points)
    prefs = np.asarray(prefs).astype('float32')
    return o1, o2, prefs, critical_points


def generate_preference_pairs_uniform(self, trajectories, critical_points, number_of_queries=200, truth=100):
    for _ in range(number_of_queries):
        segments, points = self.random_sample_batch_segments_with_critical_points(trajectories, critical_points, number_of_sampled_segments=2)
        query = self.oracle_reward_function(segments[0], segments[1], truth, points)
        self.add_pairs_with_critical_points(query[0], query[1], query[2], query[3])