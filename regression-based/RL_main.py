from Environment_RL import AutoFeature_env
from Agent_RL import Autofeature_agent

def main_RL(tables):
    # Parameters for the environment
    base_train_path = "./data/temp_train.csv"
    base_test_path = "./data/temp_test.csv"
    repo_train_path = ['./data/' + i + '_train.csv' for i in tables]
    repo_test_path = ['./data/' + i + '_test.csv' for i in tables]

    index_col = 'Day'
    target_col = 'Temperaturetemp'

    model_target = 0
    max_try_num = 4

    env = AutoFeature_env(base_train_path, base_test_path, repo_train_path, repo_test_path, index_col, target_col, model_target, max_try_num)

    # Parameters for the agent
    learning_rate = 0.05
    reward_decay = 0.9
    e_greedy = 1
    update_freq = 50
    mem_cap = 1000
    BDQN_batch_size = 3

    autodata = Autofeature_agent(env, BDQN_batch_size, learning_rate, reward_decay, e_greedy, update_freq, mem_cap, BDQN_batch_size)

    print("Agent Ready!")

    # Train the workload
    autodata.train_workload()


if __name__ == "__main__":
    tables = ['pm10_daily_summary.csv', 'pm25_frm_daily_summary.csv', 'pressure_daily_summary.csv',
              'rh_and_dp_daily_summary.csv', 'so2_daily_summary.csv', 'voc_daily_summary.csv', 'wind_daily_summary.csv'
              ]
    main_RL(tables)