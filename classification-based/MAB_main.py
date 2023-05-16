from Environment_MAB import AutoFeature_env
from Agent_MAB import AutoFeature_agent
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)

def main_MAB(tables):

    # Parameters for the environment
    base_train_path = "../data/temp_train.csv"
    base_test_path = "../data/temp_test.csv"
    repo_train_path = ['../data/' + i + '_train.csv' for i in tables]
    repo_test_path = ['../data/' + i + '_test.csv' for i in tables]

    index_col = 'Day'
    target_col = 'Temperaturetemp'

    model_target = 0.60

    max_try_num = 10

    topl = 3

    random_state = 42

    model = {'XGB': {}}

    env = AutoFeature_env(base_train_path, base_test_path, repo_train_path, repo_test_path, index_col, target_col, model_target, model, max_try_num, topl)

    res_csv = "./data/result_mab.csv"

    autofeature = AutoFeature_agent(env, res_csv, random_state)

    print("Agent Ready!")

    # Train the workload
    autofeature.augment()


if __name__ == "__main__":
    tables = ['pm10_daily_summary', 'pm25_frm_daily_summary', 'pressure_daily_summary',
              'rh_and_dp_daily_summary', 'so2_daily_summary', 'voc_daily_summary', 'wind_daily_summary'
              ]
    main_MAB(tables)
