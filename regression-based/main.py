import pandas as pd
from sklearn.model_selection import train_test_split
from donwload_data import query_daily, prepare_data_for_ml
from MAB_main import main_MAB
from RL_main import main_RL
import os

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    print("starting")

    tables = ['co_daily_summary', 'hap_daily_summary', 'lead_daily_summary', 'no2_daily_summary', 'nonoxnoy_daily_summary',
              'o3_daily_summary', 'pm10_daily_summary', 'pm25_frm_daily_summary', 'pm25_nonfrm_daily_summary',
              'pm25_speciation_daily_summary', 'pressure_daily_summary', 'rh_and_dp_daily_summary',
              'so2_daily_summary', 'temperature_daily_summary', 'voc_daily_summary', 'wind_daily_summary']
    query_daily(tables)


    for entry in ['temp'] + tables:
        print(entry)

        df = pd.read_csv(f"./data/{entry}.csv")
        df.columns = list(map(lambda x: x + entry if x != 'Day' else x, list(df.columns)))

        col_list = list(df.columns)
        if 'Day' in col_list:
            col_list.remove('Day')
        df[col_list] = prepare_data_for_ml(df[col_list])

        a_train, a_test = train_test_split(df, test_size=0.4, random_state=42)
        a_train.to_csv(f"./data/{entry}_train.csv", index=False)
        a_test.to_csv(f"./data/{entry}_test.csv", index=False)

    print("DONE")
    main_MAB(tables)
    # main_RL(tables)
