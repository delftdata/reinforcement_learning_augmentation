from Environment_MAB import AutoFeature_env
from Agent_MAB import AutoFeature_agent

def main_MAB(tables):

    # Parameters for the environment
    base_train_path = "./data/temp_train.csv"
    base_test_path = "./data/temp_test.csv"
    repo_train_path = ['./data/' + i + '_train.csv' for i in tables]
    repo_test_path = ['./data/' + i + '_test.csv' for i in tables]

    index_col = 'Day'
    target_col = 'Temperaturetemp'

    model_target = 0.60

    max_try_num = 40

    topl = 3

    random_state = 42

    env = AutoFeature_env(base_train_path, base_test_path, repo_train_path, repo_test_path, index_col, target_col, model_target, max_try_num, topl)

    res_csv = "./data/result_mab.csv"

    autofeature = AutoFeature_agent(env, res_csv, random_state)

    print("Agent Ready!")

    # Train the workload
    autofeature.augment()


if __name__ == "__main__":
    tables = ['pm10_daily_summary.csv', 'pm25_frm_daily_summary.csv', 'pressure_daily_summary.csv',
              'rh_and_dp_daily_summary.csv', 'so2_daily_summary.csv', 'voc_daily_summary.csv', 'wind_daily_summary.csv'
              ]
    main_MAB(tables)