#codingL utf8

import math
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import adjusted_mutual_info_score, mean_squared_error, roc_auc_score


class AutoFeature_env(object):
    """
    The env of AutoFeature_RL
    """
    def __init__(self, base_train_path, base_test_path, repo_train_path_list, repo_test_path_list, index_col, target_col, target, max_try_num = 5):
        self.base_train_table = pd.read_csv(base_train_path)
        self.base_test_table = pd.read_csv(base_test_path)

        self.repo_train_table_list = []
        for repo_train_path in repo_train_path_list:
            self.repo_train_table_list.append(pd.read_csv(repo_train_path))

        self.repo_test_table_list = []
        for repo_test_path in repo_test_path_list:
            self.repo_test_table_list.append(pd.read_csv(repo_test_path))

        self.index_col = index_col
        self.target_col = target_col

        self.target = target  # AUC or MSE

        self.current_training_set = None
        self.current_test_set = None

        self.current_joined_training_set = None
        self.current_joined_test_set = None

        self.current_model = None

        self.table_sel_vector = []
        self.feature_sel_vec_list = []
        self.feature_sel_vector = []
        self.feature_charac_vector = []

        self.original_score = None
        self.cur_score = None
        self.prev_score = None

        self.cur_state = None
        self.prev_state = None

        self.max_try_num = max_try_num
        self.try_num = 0

        self.total_candidate_feature_num = 0
        self.all_repo_feature = []
        for i in range(len(self.repo_train_table_list)):
            tmp_repo_table_cols = list(self.repo_train_table_list[i].columns)
            tmp_repo_table_cols.remove(self.index_col)
            self.total_candidate_feature_num += len(tmp_repo_table_cols)
            self.all_repo_feature.append(tmp_repo_table_cols)

        # Two types of action
        self.action_table = []
        self.action_table_valid = []

        self.selected_table = []

        self.action_feature = []
        self.action_feature_valid = []

        self.selected_feature = []


        self.init_env()


    def init_env(self):
        # Init training set
        self.current_joined_training_set = self.base_train_table.copy()
        self.current_joined_test_set = self.base_test_table.copy()

        self.current_training_set = self.base_train_table.copy()
        self.current_test_set = self.base_test_table.copy()

        # Init cur_state
        self.get_current_state(0)

        # Init the model
        X_train, Y_train = self.get_training_dataset()
        self.current_model = self.model_training(X_train, Y_train)

        print('-' * 20 + "Init:" + '-' * 20)
        train_auc = self.model_test_rmse(X_train, Y_train)
        print(f"Train RMSE score: {train_auc}")

        X_test, Y_test = self.get_test_dataset()
        test_auc = self.model_test_rmse(X_test, Y_test)
        print(f"Test RMSE Score: {test_auc}")

        self.cur_score = test_auc
        self.original_score = test_auc

        # Check the valid action
        self.action_table = [_ for _ in range(len(self.repo_train_table_list))]
        self.action_table_valid = [_ for _ in self.action_table]
        self.selected_table = []

        for i in range(len(self.all_repo_feature)):
            for j in range(len(self.all_repo_feature[i])):
                self.action_feature.append([i,j])

        self.action_feature_valid = []
        self.selected_feature = []
        self.generate_valid_feature_action()

        self.try_num = 0

    def reset(self):
        # Init training set
        self.current_joined_training_set = self.base_train_table.copy()
        self.current_joined_test_set = self.base_test_table.copy()

        self.current_training_set = self.base_train_table.copy()
        self.current_test_set = self.base_test_table.copy()

        # Init the model
        X_train, Y_train = self.get_training_dataset()
        self.current_model = self.model_training(X_train, Y_train)

        X_test, Y_test = self.get_test_dataset()
        test_auc = self.model_test_rmse(X_test, Y_test)

        print('-' * 20 + "Reset:" + '-' * 20)
        print(f"Test RMSE Score: {test_auc}")

        self.cur_score = test_auc

        # Check the valid action
        self.action_table_valid = [_ for _ in self.action_table]
        self.selected_table = []

        self.action_feature_valid = []
        self.generate_valid_feature_action()
        self.selected_feature = []

        self.try_num = 0



    def step(self, action):
        """
        Execute the action
        :param action: the action selected by the agent
        :return: reward, done or not
        """
        print(f"Action: {action}")

        if action[0] == 't':
            true_action = self.action_table[action[1]]

            self.current_joined_training_set = pd.merge(self.current_training_set, self.repo_train_table_list[true_action], how = 'left', on=self.index_col)
            self.current_joined_test_set = pd.merge(self.current_test_set, self.repo_test_table_list[true_action], how='left', on = self.index_col)

            X_train = self.current_joined_training_set.drop([self.index_col, self.target_col], axis=1)
            Y_train = self.current_joined_training_set[self.target_col]
            self.current_model = self.model_training(X_train, Y_train)
            X_test = self.current_joined_test_set.drop([self.index_col, self.target_col], axis=1)
            Y_test = self.current_joined_test_set[self.target_col]
            test_rmse = self.model_test_rmse(X_test, Y_test)

            # Update the reward and the valid action
            self.action_table_valid.remove(action[1])
            self.selected_table.append(true_action)
            self.add_valid_feature_action(true_action)

            self.try_num += 1
            self.prev_state = self.cur_state
            self.get_current_state(1)

            self.prev_score = self.cur_score
            self.cur_score = test_rmse



            if self.try_num > self.max_try_num:
                print("Try too much times!!!")
                done = True
                return self.cur_state, self.prev_score - self.cur_score, done
            else:
                done = False
                return self.cur_state, self.prev_score - self.cur_score, done

        elif action[0] == 'f':
            true_action = self.action_feature[action[1]]
            selected_table_cols = list(self.repo_train_table_list[true_action[0]].columns)
            selected_table_cols.remove(self.index_col)

            # Add new features
            tmp_repo_train_table = self.repo_train_table_list[true_action[0]].loc[:, [self.index_col, selected_table_cols[true_action[1]]]]
            tmp_repo_test_table = self.repo_test_table_list[true_action[0]].loc[:, [self.index_col, selected_table_cols[true_action[1]]]]

            self.current_training_set = pd.merge(self.current_training_set, tmp_repo_train_table, how='left', on=self.index_col)
            self.current_test_set = pd.merge(self.current_test_set, tmp_repo_test_table, how='left', on=self.index_col)

            # Train and test on new dataset
            X_train, Y_train = self.get_training_dataset()
            self.current_model = self.model_training(X_train, Y_train)

            X_test, Y_test = self.get_test_dataset()
            test_rmse = self.model_test_rmse(X_test, Y_test)

            # return
            self.action_feature_valid.remove(action[1])
            self.selected_feature.append(action[1])

            self.try_num += 1
            self.prev_state = self.cur_state
            self.get_current_state(2)

            self.prev_score = self.cur_score
            self.cur_score = test_rmse

            if self.try_num > self.max_try_num:
                print("Try too much times!!!")
                done = True
                return self.cur_state, self.prev_score - self.cur_score, done
            else:
                done = False
                return self.cur_state, self.prev_score - self.cur_score, done


    def get_training_dataset(self):
        X_train = self.current_training_set.drop([self.index_col, self.target_col], axis = 1)
        Y_train = self.current_training_set[self.target_col]
        return X_train, Y_train

    def get_test_dataset(self):
        X_test = self.current_test_set.drop([self.index_col, self.target_col], axis = 1)
        Y_test = self.current_test_set[self.target_col]
        return X_test, Y_test

    def model_training(self, X_train, Y_train):
        new_model = XGBRegressor(use_label_encoder=False)
        new_model.fit(X_train, Y_train, eval_metric='rmse')
        return new_model

    def model_test_rmse(self, X_test, Y_test):
        y_test_pred = self.current_model.predict(X_test)

        rmse_score = mean_squared_error(Y_test, y_test_pred)
        return rmse_score

    def get_current_state(self, update_type):
        """
        Update the state representation
        :param update_type: 0-init, 1-table, 2-feature
        :return:
        """
        if update_type == 0:
            # Table vector
            self.table_sel_vector = [0 for _ in range(len(self.repo_train_table_list))]
            for tbl_id in self.selected_table:
                self.table_sel_vector[tbl_id] = 1

            # Feature vector
            self.feature_sel_vec_list = []

            for i in range(len(self.all_repo_feature)):
                one_repo_feature_vec = [0 for _ in range(len(self.all_repo_feature[i]))]
                self.feature_sel_vec_list.append(one_repo_feature_vec)

            for action in range(len(self.selected_feature)):
                self.feature_sel_vec_list[action[0]][action[1]] = 1

            self.feature_sel_vector = []
            for vec in self.feature_sel_vec_list:
                self.feature_sel_vector.extend(vec)

            # Feature characteristics
            self.feature_charac_vector = []
            for i in range(len(self.all_repo_feature)):
                one_repo_feature_charac_vec = [[0,0,0] for _ in range(len(self.all_repo_feature[i]))]
                self.feature_charac_vector.append(one_repo_feature_charac_vec)


        elif update_type == 1:
            # Table vector
            self.table_sel_vector[self.selected_table[-1]] = 1

            selected_table_cols = list(self.repo_train_table_list[self.selected_table[-1]].columns)
            selected_table_cols.remove(self.index_col)

            for i in range(len(selected_table_cols)):
                # Variance
                cha_vari = self.current_joined_training_set[selected_table_cols[i]].values.var()

                # PCC
                covar = self.current_joined_training_set[selected_table_cols[i]].cov(self.current_joined_training_set[self.target_col])
                var_1 = self.current_joined_training_set[selected_table_cols[i]].var()
                var_2 = self.current_joined_training_set[self.target_col].var()

                if var_1 * var_2 == 0:
                    cha_pcc = 0
                else:
                    cha_pcc = covar / math.sqrt(var_1 * var_2)

                # MI
                cha_mi = adjusted_mutual_info_score(self.current_joined_training_set[selected_table_cols[i]].fillna(-1).values,
                                                    self.current_joined_training_set[self.target_col].values)

                self.feature_charac_vector[self.selected_table[-1]][i][0] = cha_vari
                self.feature_charac_vector[self.selected_table[-1]][i][1] = cha_pcc
                self.feature_charac_vector[self.selected_table[-1]][i][2] = cha_mi

        elif update_type == 2:
            action_pos = self.selected_feature[-1]
            self.feature_sel_vec_list[self.action_feature[action_pos][0]][self.action_feature[action_pos][1]] = 1

            self.feature_sel_vector = []
            for vec in self.feature_sel_vec_list:
                self.feature_sel_vector.extend(vec)

        self.cur_state = [self.table_sel_vector, self.feature_sel_vector, self.feature_charac_vector]



    def get_current_features(self):
        cur_train_set_col = list(self.current_training_set.columns)
        cur_train_set_col.remove(self.index_col)
        cur_train_set_col.remove(self.target_col)
        return len(cur_train_set_col)

    def generate_valid_feature_action(self):
        for repo_table_id in self.selected_table:
            tmp_repo_table_cols = list(self.repo_train_table_list[repo_table_id].columns)
            tmp_repo_table_cols.remove(self.index_col)
            for j in range(len(tmp_repo_table_cols)):
                action = self.action_feature.index([repo_table_id, j])
                self.action_feature_valid.append(action)

    def add_valid_feature_action(self, new_table_id):
        tmp_repo_table_cols = list(self.repo_train_table_list[new_table_id].columns)
        tmp_repo_table_cols.remove(self.index_col)
        for j in range(len(tmp_repo_table_cols)):
            action = self.action_feature.index([new_table_id, j])
            self.action_feature_valid.append(action)

    def get_table_action_len(self):
        return len(self.action_table)

    def get_valid_table_action_len(self):
        return len(self.action_table_valid)

    def get_feature_action_len(self):
        return len(self.action_feature)

    def get_valid_feature_action_len(self):
        return len(self.action_feature_valid)

    def get_action_len(self):
        return len(self.action_table) + len(self.action_feature)

    def get_state_len(self):
        return len(self.action_table) + len(self.action_feature) + 3 * len(self.action_feature)