#coding: utf8
import random
import math
import time
import csv
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import os
import copy


class Net(nn.Module):
    """
    Branch DQN
    """

    def __init__(self, input_num, table_action_num, feature_action_num):
        """
        Init the parameters of the Q-Network
        """
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_num, 2 ** 12)
        self.fc1.weight.data.normal_(0, 0.01)

        self.fc2 = nn.Linear(2 ** 12, 2 ** 10)
        self.fc2.weight.data.normal_(0, 0.01)

        self.tbl_fc1 = nn.Linear(2 ** 10, 2 ** 8)
        self.tbl_fc1.weight.data.normal_(0, 0.01)

        self.tbl_fc2 = nn.Linear(2 ** 8, table_action_num)
        self.tbl_fc2.weight.data.normal_(0, 0.01)

        self.ft_fc1 = nn.Linear(2 ** 10, 2 ** 8)
        self.ft_fc1.weight.data.normal_(0, 0.01)

        self.ft_fc2 = nn.Linear(2 ** 8, feature_action_num)
        self.ft_fc2.weight.data.normal_(0, 0.01)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        tbl_x = self.tbl_fc1(x)
        ft_x = self.ft_fc1(x)

        tbl_x = self.tbl_fc2(tbl_x)
        ft_x = self.ft_fc2(ft_x)

        action_value = torch.concat([tbl_x, ft_x], 1)
        return action_value


class Autofeature_agent(object):
    """
    Agent of Autofeature
    """
    def __init__(self, env, bdqn_csv, learning_rate = 0.05, reward_decay = 0.9, e_greedy = 1, update_freq = 50, mem_cap=100000, BDQN_batch_size = 32):
        self.env = env
        self.res_csv = bdqn_csv
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.batch_size = BDQN_batch_size

        self.learning_step_counter = 0
        self.update_freq = update_freq

        self.mem_counter = 0
        self.memory_capacity = mem_cap
        self.mem = np.zeros((self.memory_capacity, self.env.get_state_len() * 2 + self.env.get_action_len() + 1))

        # print("Choose device")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # self.device = torch.device("cuda")
        # print("Device: " + str(self.device) + " will be used!")
        self.device = torch.device("cpu")

        self.eval_net = Net(self.env.get_state_len(), self.env.get_table_action_len(), self.env.get_feature_action_len()).to(self.device)
        self.target_net = Net(self.env.get_state_len(), self.env.get_table_action_len(), self.env.get_feature_action_len()).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self):
        """
        Choose a new action
        :return:
        """
        if np.random.uniform() > self.epsilon:
            print("Chosen By Network")
            x = self.state_representation(self.env.cur_state)

            x = torch.unsqueeze(torch.Tensor(x), 0).to(self.device)
            actions_value_tmp = self.eval_net.forward(x)
            action_valid = []
            action_valid.extend(self.env.action_table_valid)
            merge_feature_action = [_+self.env.get_table_action_len() for _ in self.env.action_feature_valid]
            action_valid.extend(merge_feature_action)
            actions_value = actions_value_tmp[0, action_valid]

            max_action_val = torch.max(actions_value)
            y = torch.ones(1, len(actions_value)) * -10000
            y = y.to(self.device)
            max_actions = torch.where(actions_value == max_action_val, actions_value, y)
            candidate_action = []
            for i in range(len(actions_value)):
                if max_actions[0, i] != -10000:
                    candidate_action.append(i)

            if len(candidate_action) > 1:
                action_num = action_valid[candidate_action[np.random.randint(len(candidate_action))]]
            elif len(candidate_action) == 0:
                action_num = action_valid[np.random.randint(len(action_valid))]
            else:
                action_num = action_valid[candidate_action[0]]
        else:
            print("Chosen By Random")
            action_type = 0
            if (self.env.get_valid_table_action_len() > 0 and self.env.get_valid_feature_action_len() > 0):
                action_type = np.random.randint(2)
            elif (self.env.get_valid_table_action_len() > 0 and self.env.get_valid_feature_action_len() == 0):
                action_type = 0
            elif (self.env.get_valid_table_action_len() == 0 and self.env.get_valid_feature_action_len() > 0):
                action_type = 1


            if action_type == 0:
                action_num = self.env.action_table_valid[np.random.randint(self.env.get_valid_table_action_len())]
            else:
                action_num = self.env.action_feature_valid[np.random.randint(self.env.get_valid_feature_action_len())] + self.env.get_table_action_len()

        if action_num < len(self.env.action_table):
            return ['t', action_num]
        else:
            return ['f', action_num - len(self.env.action_table)]


    def store_transition(self, s, a, r, s_):
        if a[0] == 't':
            action_pos = a[1]
        elif a[0] == 'f':
            action_pos = a[1] + self.env.get_table_action_len()

        action_vec = [0 for _ in range(self.env.get_action_len())]

        action_vec[action_pos] = 1

        s_seq = self.state_representation(s)
        s__seq = self.state_representation(s_)

        trans = np.hstack((s_seq, action_vec, [r], s__seq))
        index = self.mem_counter % self.memory_capacity
        self.mem[index, :] = trans
        self.mem_counter += 1


    def learn(self):
        sample_index = np.random.choice(min(self.mem_counter, self.memory_capacity), self.batch_size)
        b_memory = self.mem[sample_index, :]
        state_len = self.env.get_state_len()
        b_s = torch.tensor(b_memory[:, :state_len], dtype=torch.float).to(self.device)
        b_a = torch.tensor(b_memory[:, state_len: state_len + self.env.get_action_len()], dtype=torch.long)
        b_r = torch.tensor(
            b_memory[:, state_len + self.env.get_action_len(): state_len + self.env.get_action_len() + 1],
            dtype=torch.float).to(self.device)
        b_s_ = torch.tensor(b_memory[:, -state_len:], dtype=torch.float).to(self.device)

        b_a_one_action = torch.zeros(b_a.size()[0], 1, dtype=torch.long)
        action_index = np.argwhere(b_a.numpy() > 0)

        for i in range(len(action_index)):
            b_a_one_action[i][0] = action_index[i][1]

        b_a_one_action = b_a_one_action.to(self.device)

        q_eval = self.eval_net(b_s).gather(1, b_a_one_action)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learning_step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step_counter += 1


    def train_workload(self):
        with open('information_rl.csv', 'w') as file:
            file.write('iteration, accuracy\n')
            X_test, Y_test = self.env.get_test_dataset()
            test_mse = self.env.model_test_rmse(X_test, Y_test)
            file.write(f'0, {test_mse}\n')

            episode_num = 40
            cut_off = 30
            const_a = math.pow(0.001, 1 / episode_num)

            time_start = time.time()
            for episode in range(episode_num):
                self.env.reset()

                cur_state = copy.deepcopy(self.env.cur_state)

                counter = 0
                while True:
                    action = self.choose_action()
                    state_next, reward, done = self.env.step(action)
                    if done:
                        counter += 1

                        X_train, Y_train = self.env.get_training_dataset()
                        self.env.current_model = self.env.model_training(X_train, Y_train)
                        X_test, Y_test = self.env.get_test_dataset()
                        test_mse = self.env.model_test_rmse(X_test, Y_test)
                        file.write(f'{counter}, {test_mse}\n')

                        time_end = time.time()

                        print("Epsilon:" + str(self.epsilon))
                        print("Schema：" + str(episode))
                        # print("The selected data are：")
                        # for each_cluster_action in action:
                        #     print("Cluster：" + str(each_cluster_action[0]) + ", Ratio: " + str(each_cluster_action[1]))
                        print(f"The number of features in current training set：{self.env.get_current_features()}")
                        print("The RMSE of current model：" + str(self.env.cur_score))
                        print("Benefit：" + str(max(0, self.env.original_score - self.env.cur_score)))
                        print("Time：" + str(time_end - time_start))

                        break

                    self.store_transition(cur_state, action, reward, state_next)
                    if self.mem_counter > self.batch_size:
                        self.learn()

                    cur_state = state_next

                if self.epsilon >= episode_num - cut_off:
                    self.epsilon = 0.0001
                else:
                    self.epsilon = math.pow(const_a, episode)

            file.write(str([episode, self.epsilon, self.env.get_current_features(), self.env.cur_score,
                            max(0, self.env.prev_score - self.env.cur_score), time_end - time_start]))



    def state_representation(self, state):
        """
        Get the state vector
        :return:
        """
        table_vector = torch.Tensor(state[0])
        feature_vector = torch.Tensor(state[1])

        charac_list = []

        for i in range(len(state[2])):
            for j in range(len(state[2][i])):
                charac_list.extend(state[2][i][j])

        charac_vector = torch.Tensor(charac_list)

        state_vector = torch.concat([table_vector, feature_vector,charac_vector], 0)

        return state_vector


