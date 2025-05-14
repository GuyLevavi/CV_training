import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from basic.python.azrieli_code_guy.summary_table import summary_df

DATA_PATH = r'C:\Users\user\Desktop\CV Training Guy\CV_training\basic\python\Basic - Azrieli & Sons data'


def calc_nan_df(df):
    nan_summary = df.isna().sum()
    nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)
    nan_summary = nan_summary.reset_index()
    nan_summary.columns = ['Column', 'NaN Count']
    nan_summary['NaN Ratio'] = np.round(nan_summary['NaN Count'] / df.shape[0], 2)
    return nan_summary


def load_drivers_data():
    """
    Load the old drivers data from CSV files.
    """
    old_drivers_df = pd.read_csv(os.path.join(DATA_PATH, 'Drivers_with_kviut.csv'))
    new_drivers_df = pd.read_csv(os.path.join(DATA_PATH, 'new_drivers.csv'))
    new_drivers_df['vetek'] = new_drivers_df['vetek'].apply(lambda x: x / 365.25)
    # concat
    drivers_df = pd.concat([old_drivers_df, new_drivers_df])
    drivers_df = drivers_df.iloc[:, 1:]
    drivers_df.rename(columns={'id': 'driver_id'}, inplace=True)
    drivers_df['gender'].replace(['m', 'male', 'boy'], 'M', inplace=True)
    drivers_df['gender'].replace(['girl', 'female', 'woman'], 'F', inplace=True)
    drivers_df['gender'].replace(['none', 'unknown'], '?', inplace=True)
    drivers_df['gender'].fillna('?', inplace=True)
    drivers_df['birthdate'] = drivers_df['birthdate'].astype("datetime64[ns]")
    # drivers_df['birthdate'] = pd.to_datetime(drivers_df['birthdate'], errors='coerce')
    # TODO: handle NaNs in birthdate, and 1.1.1900
    return drivers_df


def load_salary_data():
    """
    Load the salary data from CSV files.
    """
    salary_df = pd.read_csv(os.path.join(DATA_PATH, 'taarif.csv'))
    salary_df = salary_df.iloc[1:, :]
    salary_df.replace({
        'telecommunication_ltd': 'hot',
        'dbs': 'yes',
        'mizranei_kfar_saba': 'aminach'
    }, inplace=True)
    salary_df['weekend_bonus'] = salary_df['weekend_bonus'] / 100
    salary_df['night_bonus'] = salary_df['night_bonus'] / 100
    # TODO: fill electricity and bituch leumi
    return salary_df


def calculate_weekend_and_night_ratios(row):
    total_duration = row["end_time"] - row["start_time"]
    total_seconds = total_duration.total_seconds()
    weekend_seconds = 0
    night_seconds = 0

    # Iterate through each hour in the date range
    current_time = row["start_time"]
    while current_time < row["end_time"]:
        next_hour = current_time + timedelta(hours=1)

        # Check if this hour is a weekend (Friday or Saturday)
        if current_time.weekday() in [4, 5]:
            weekend_seconds += min((next_hour - current_time).total_seconds(),
                                   (row["end_time"] - current_time).total_seconds())

        # Check if this hour is during the night (22:00 to 05:00)
        if (22 <= current_time.hour or current_time.hour < 5):
            night_seconds += min((next_hour - current_time).total_seconds(),
                                 (row["end_time"] - current_time).total_seconds())

        current_time = next_hour

    weekend_ratio = weekend_seconds / total_seconds
    night_ratio = night_seconds / total_seconds

    return pd.Series([weekend_ratio, night_ratio])


def load_trips_data(recalculate=False):
    """
    Load the trips data from CSV files.
    """
    if recalculate:
        trips_data_list = os.listdir(os.path.join(DATA_PATH, 'trips_data'))
        trips_data_no_dups_list = [trip for trip in trips_data_list if '(2)' not in trip]
        trips_dfs = [pd.read_csv(os.path.join(DATA_PATH, 'trips_data', trip)) for trip in trips_data_no_dups_list]
        trips_df = pd.concat(trips_dfs, ignore_index=True)
        trips_df = trips_df.iloc[:, 1:]
        trips_df['start_time'] = trips_df['start_time'].astype("datetime64[ns]")
        trips_df['end_time'] = trips_df['end_time'].astype("datetime64[ns]")
        trips_df[['weekend_ratio', 'night_ratio']] = trips_df.apply(calculate_weekend_and_night_ratios, axis=1)
        trips_df.to_csv(os.path.join(DATA_PATH, 'trips_data', 'trips_data.csv'), index=False)
    else:
        trips_df = pd.read_csv(os.path.join(DATA_PATH, 'trips_data', 'trips_data.csv'))
        trips_df['start_time'] = trips_df['start_time'].astype("datetime64[ns]")
        trips_df['end_time'] = trips_df['end_time'].astype("datetime64[ns]")
        trips_df['start_month'] = trips_df['start_time'].dt.month
        trips_df['start_month_year'] = trips_df['start_time'].dt.to_period('M')
        # trips_df['end_month'] = trips_df['end_time'].dt.month

    return trips_df


def compute_salary(km, base, extra,
                   night_multiplier, night_ratio,
                   weekend_multiplier, week_ratio):
    # TODO: is extra alone or added to base?
    return (min(km, 200) * base + max(km - 200, 0) * extra) * (
            1 + night_multiplier * night_ratio + weekend_multiplier * week_ratio)


if __name__ == '__main__':
    drivers_df = load_drivers_data()
    salary_df = load_salary_data()
    trips_df = load_trips_data()
    trips_salary_df = pd.merge(trips_df, salary_df, on='customer')
    # TODO: remove when figuring out what to do with NaNs
    trips_salary_df.dropna(inplace=True)
    # TODO: salary seems quite high, verify
    trips_salary_df['salary'] = trips_salary_df.apply(
        lambda row: compute_salary(row['km'], row['basic_taarif'], row['extra_milage'],
                                   row['night_bonus'], row['night_ratio'],
                                   row['weekend_bonus'], row['weekend_ratio']), axis=1)
    drivers_salary_df = trips_salary_df[['driver_id', 'salary', 'start_month_year', 'km']] \
        .groupby(['driver_id', 'start_month_year']) \
        .agg(total_income=('salary', 'sum'),
             total_km=('km', 'sum')) \
        .reset_index()
    summary_df = pd.merge(drivers_df, drivers_salary_df, on='driver_id')
    summary_df['age'] = (summary_df['start_month_year'].dt.to_timestamp() - summary_df['birthdate']).dt.days / 365.25
    summary_df[['total_km', 'total_income']] = summary_df[['total_km', 'total_income']].round(0).astype(int)
    summary_df[['age', 'vetek']] = summary_df[['age', 'vetek']].round(1)
    summary_df.drop(columns=['birthdate'], inplace=True)
    summary_df.rename(columns={'start_month_year': 'month'}, inplace=True)
    summary_df = summary_df[['driver_id','month','total_income','total_km','gender','age','vetek']]
    summary_df = summary_df.set_index(['driver_id', 'month']).sort_index()
