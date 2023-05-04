import pandas as pd
from sklearn.model_selection import train_test_split
from donwload_data import query_daily, prepare_data_for_ml
from MAB_main import main_MAB
from RL_main import main_RL
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    current_dir = os.getcwd()
    final_dir = os.path.join(current_dir, r'data')
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    print("starting")

    final_dir = os.path.join(os.getcwd(), r'data3')

    tables = os.listdir(final_dir)

    for entry in tables:
        print(entry)
        df = pd.read_csv(f"./data3/{entry}")
        df.columns = list(map(lambda x: str(x).upper(), list(df.columns)))
        df.columns = list(map(lambda x: entry + x if x != 'DBN' else x, list(df.columns)))

        col_list = list(df.columns)
        if 'DBN' in col_list:
            col_list.remove('DBN')
        df[col_list] = prepare_data_for_ml(df[col_list])

        a_train, a_test = train_test_split(df, test_size=0.4, random_state=42)
        a_train.to_csv(f"./data3/{entry}_train.csv", index=False)
        a_test.to_csv(f"./data3/{entry}_test.csv", index=False)

    # query_daily()

    # for query_job in ['codaily', 'pm25daily',
    #                   'no2daily', 'o3daily', 'temp']:
    #     print(query_job)
    #
    #     df = pd.read_csv(f"./data/{query_job}.csv")
    #
    #     a_train, a_test = train_test_split(df, test_size=0.4, random_state=42)
    #
    #     a_train.to_csv(f"./data/{query_job}_train.csv", index=False)
    #
    #     a_test.to_csv(f"./data/{query_job}_test.csv", index=False)

    tables.remove('base.csv')
    print("DONE")
    # main_MAB(tables)
    main_RL(tables)
