import pandas as pd
from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def prepare_data_for_ml(dataframe):
    df = dataframe.fillna(0)
    df = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == object else x)

    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(df)
    normalized_X = pd.DataFrame(scaled_X, columns=df.columns)

    return normalized_X


# def main():
#     for table in ['o3_daily_summary', 'pm10_daily_summary', 'pm25_frm_daily_summary', 'pressure_daily_summary',
#                   'rh_and_dp_daily_summary', 'so2_daily_summary', 'voc_daily_summary', 'wind_daily_summary']:
#         print(table)
#         df = pd.read_csv(f'./data/air/{table}.csv', index_col='id')
#
#         new_df = prepare_data_for_ml(df)
#
#         new_df.index.name = 'id'
#         new_df.to_csv(f'./data/encoded/{table}.csv')


def query_daily(tables):
    limit = 50000

    QUERYtemp = f"""
        SELECT
           T.date_local AS Day, AVG(T.arithmetic_mean) AS Temperature, AVG(first_max_value), AVG(first_max_hour), AVG(observation_count), AVG(observation_percent), ANY_VALUE(state_name), ANY_VALUE(county_name), ANY_VALUE(city_name)
        FROM
          `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary` as T
          
        GROUP BY Day
        ORDER BY Day

        LIMIT {limit}
    """
    client = bigquery.Client.from_service_account_json('key.json')
    df_temp = client.query(QUERYtemp).to_dataframe().set_index('Day')

    df_temp.index.name = 'Day'
    df_temp.to_csv(f"./data/temp.csv")

    for table_name in tables:
        print(table_name)

        query_table = f"""
            SELECT
               o3.date_local AS Day, AVG(o3.aqi) AS o3_AQI, AVG(first_max_value), AVG(first_max_hour), AVG(observation_count), AVG(observation_percent), ANY_VALUE(state_name), ANY_VALUE(county_name), ANY_VALUE(city_name)
            FROM
              `bigquery-public-data.epa_historical_air_quality.{table_name}` as o3
    
            GROUP BY Day
            ORDER BY Day
    
            LIMIT {limit}
        """

        client = bigquery.Client.from_service_account_json('key.json')
        df_temp = client.query(query_table).to_dataframe().set_index('Day')

        df_temp.index.name = 'Day'
        df_temp.to_csv(f"./data/{table_name}.csv")


if __name__ == "__main__":

    query_daily()

    for query_job in ['codaily', 'pm25daily',
                      'no2daily', 'o3daily', 'temp']:
        print(query_job)

        df = pd.read_csv(f"./data/{query_job}.csv")

        a_train, a_test = train_test_split(df, test_size=0.2, random_state=42)

        a_train.to_csv(f"./data/{query_job}_train.csv", index=False)

        a_test.to_csv(f"./data/{query_job}_test.csv", index=False)
