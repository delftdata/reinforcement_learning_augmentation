#coding: utf8
import random
import math
import time
import csv

class AutoFeature_agent(object):
    """
    The agent of AutoFeature
    """

    def __init__(self, env, mab_csv, random_state):
        self.env = env
        self.mab_result_csv = mab_csv
        self.random_state = random_state
        self.gamma = 0.6

        self.ucb_score_list = [0 for _ in range(len(self.env.repo_train_table_list))]
        self.action_num_list = [0 for _ in range(len(self.env.repo_train_table_list))]
        self.acc_reward_list = [0 for _ in range(len(self.env.repo_train_table_list))]

    def choose_action(self):
        """
        Choose a new action
        :return: the action chosen for next step
        """
        ucb_tmp_list = []
        for i in range(len(self.ucb_score_list)):
            if i in self.env.action_valid:
                ucb_tmp_list.append([i, self.ucb_score_list[i]])

        sort_ucb_tmp_list = sorted(ucb_tmp_list, key=lambda x: x[1], reverse=True)


        max_ucb_list = [0]

        for i in range(1,len(sort_ucb_tmp_list)):
            if sort_ucb_tmp_list[i][1] == sort_ucb_tmp_list[0][1]:
                max_ucb_list.append(i)

        if len(max_ucb_list) > 1:
            pos = max_ucb_list[random.randint(0, len(max_ucb_list) - 1)]
        else:
            pos = 0

        action = sort_ucb_tmp_list[pos][0]
        return action


    def update_ucb(self, action_index, step_reward):
        """
        update acc_reward and ucb score
        :param action: chosen action
        :param step_reward: reward in this step
        :return:
        """
        self.acc_reward_list[action_index] = (self.acc_reward_list[action_index] * self.action_num_list[action_index] + step_reward) / (self.action_num_list[action_index] + 1)
        self.action_num_list[action_index] += 1

        self.ucb_score_list[action_index] = self.acc_reward_list[action_index] + self.gamma * math.sqrt(2 * sum(self.action_num_list)) / (self.action_num_list[action_index] + 1)


    def augment(self):
        self.env.reset()
        time_start = time.time()

        with open('information_mab.csv', 'w') as file:
            file.write('iteration, accuracy\n')
            X_test, Y_test = self.env.get_test_dataset()
            test_mse = self.env.model_test_rmse(X_test, Y_test)

            file.write(f'0, {test_mse}\n')

            counter = 0
            while True:
                action = self.choose_action()
                reward, test_auc, done = self.env.step(action)

                counter += 1

                X_test, Y_test = self.env.get_test_dataset()
                test_mse = self.env.model_test_rmse(X_test, Y_test)
                file.write(f'{counter}, {test_mse}\n')

                self.update_ucb(action, reward)

                if done:
                    time_end = time.time()

                    print(f"The number of features in current training set：{self.env.get_current_features()}")
                    print("The RMSE of current model：" + str(self.env.cur_score))
                    print("Benefit：" + str(max(0, self.env.original_score - self.env.cur_score)))
                    print("Time：" + str(time_end - time_start))

                    file.write(str(
                        [self.env.get_current_features(), self.env.cur_score, max(0, self.env.original_score - self.env.cur_score),
                         time_end - time_start]))
                    break