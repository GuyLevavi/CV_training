import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
    drivers_df = pd.concat([old_drivers_df, new_drivers_df])
    drivers_df = drivers_df.iloc[:, 1:]
    drivers_df.rename(columns={'id': 'driver_id'}, inplace=True)
    drivers_df['gender'].replace(['m', 'male', 'boy'], 'M', inplace=True)
    drivers_df['gender'].replace(['girl', 'female', 'woman'], 'F', inplace=True)
    drivers_df['gender'].replace(['none', 'unknown'], '?', inplace=True)
    drivers_df['gender'].fillna('?', inplace=True)
    drivers_df['birthdate'] = drivers_df['birthdate'].astype("datetime64[ns]")
    drivers_df.loc[drivers_df['birthdate'] == datetime(1900, 1, 1), 'birthdate'] = pd.NaT
    return drivers_df


def load_salary_data():
    """
    Load the salary data from CSV files.
    """
    salary_df = pd.read_csv(os.path.join(DATA_PATH, 'taarif_fixed.csv'))
    salary_df = salary_df.iloc[1:, :]
    salary_df.replace({
        'telecommunication_ltd': 'hot',
        'dbs': 'yes',
        'mizranei_kfar_saba': 'aminach'
    }, inplace=True)
    salary_df['weekend_bonus'] = salary_df['weekend_bonus'] / 100
    salary_df['night_bonus'] = salary_df['night_bonus'] / 100
    return salary_df


def calculate_weekend_and_night_ratios(row):
    if pd.isna(row['end_time']) or pd.isna(row['start_time']):
        return pd.Series([0, 0])
    total_duration = row["end_time"] - row["start_time"]
    total_seconds = total_duration.total_seconds()
    weekend_seconds = 0
    night_seconds = 0

    # Iterate through each hour in the date range
    current_time = row["start_time"]
    while current_time < row["end_time"]:
        next_hour = current_time + timedelta(hours=1)

        # Check if this hour is a weekend (Friday or Saturday)
        if (((current_time.weekday() == 4) and (16 <= current_time.hour)) or
                ((current_time.weekday() == 5) and (current_time.hour <= 20))):
            weekend_seconds += min((next_hour - current_time).total_seconds(),
                                   (row["end_time"] - current_time).total_seconds())

        # Check if this hour is during the night (22:00 to 06:00)
        if (22 <= current_time.hour <= 6):
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
        months = [x.split(' ')[0] for x in trips_data_no_dups_list]
        trips_df_by_month = {}
        for trip, month in zip(trips_data_no_dups_list, months):
            if month not in trips_df_by_month:
                trips_df_by_month[month] = []
            trips_df_by_month[month].append(pd.read_csv(os.path.join(DATA_PATH, 'trips_data', trip)))
        trips_df_by_month = {month: pd.concat(trips_df_by_month[month], ignore_index=True)
                             for month in np.unique(months)}
        for month, df in trips_df_by_month.items():
            df['month_year'] = pd.to_datetime(month + ' 2015', format='%B %Y')  # .dt.to_period('M')
        trips_df = pd.concat(trips_df_by_month.values(), ignore_index=True)
        trips_df = trips_df.iloc[:, 1:]
        trips_df['start_time'] = trips_df['start_time'].astype("datetime64[ns]")
        trips_df['end_time'] = trips_df['end_time'].astype("datetime64[ns]")
        trips_df[['weekend_ratio', 'night_ratio']] = trips_df.apply(calculate_weekend_and_night_ratios, axis=1)
        median = trips_df['km'].median()
        mad = (trips_df['km'] - median).abs().median()
        threshold = 3
        trips_df[abs(trips_df['km'] - median) > threshold * mad] = median
        trips_df.to_csv(os.path.join(DATA_PATH, 'trips_data.csv'), index=False)
    else:
        trips_df = pd.read_csv(os.path.join(DATA_PATH, 'trips_data.csv'))

    return trips_df


def compute_salary(km, base, extra,
                   night_multiplier, night_ratio,
                   weekend_multiplier, week_ratio, extra_alone=False):
    final_extra = extra if extra_alone else base + extra
    return (min(km, 200) * base + max(km - 200, 0) * final_extra) * (
            1 + night_multiplier * night_ratio + weekend_multiplier * week_ratio)


def compute_summary_df(drivers_df, salary_df, trips_df, extra_alone=False):
    trips_salary_df = pd.merge(trips_df, salary_df, on='customer')
    # TODO: salary seems quite high, verify
    trips_salary_df['salary'] = trips_salary_df.apply(
        lambda row: compute_salary(row['km'], row['basic_taarif'], row['extra_milage'],
                                   row['night_bonus'], row['night_ratio'],
                                   row['weekend_bonus'], row['weekend_ratio'], extra_alone), axis=1)
    drivers_salary_df = trips_salary_df[['driver_id', 'salary', 'month_year', 'km']] \
        .groupby(['driver_id', 'month_year']) \
        .agg(total_income=('salary', 'sum'),
             total_km=('km', 'sum')) \
        .reset_index()
    summary_df = pd.merge(drivers_df, drivers_salary_df, on='driver_id')
    summary_df['month_year'] = pd.to_datetime(summary_df['month_year'])  # .dt.to_period('M')
    summary_df['age'] = (summary_df['month_year'] - summary_df['birthdate']).dt.days / 365.25
    ref_date = pd.to_datetime('2015-01-01')
    summary_df['vetek'] = summary_df['vetek'] + (summary_df['month_year'] - ref_date).dt.total_seconds() / (
            365.25 * 24 * 3600)
    summary_df[['total_km', 'total_income']] = summary_df[['total_km', 'total_income']].round(0).astype(int)
    summary_df[['age', 'vetek']] = summary_df[['age', 'vetek']].round(1)
    summary_df.drop(columns=['birthdate'], inplace=True)
    summary_df.rename(columns={'month_year': 'month'}, inplace=True)
    summary_df['month'] = summary_df['month'].dt.strftime('%Y-%m')
    summary_df = summary_df[['driver_id', 'month', 'total_income', 'total_km', 'gender', 'age', 'vetek']]
    # summary_df['age'] = summary_df['age'].fillna(summary_df['age'].mean())
    summary_df = summary_df.set_index(['driver_id', 'month']).sort_index()
    save_summary(summary_df)
    return summary_df


def save_summary(summary_df, name='summary_df'):
    summary_df.to_csv(os.path.join(DATA_PATH, f'{name}.csv'))
    summary_df.to_excel(os.path.join(DATA_PATH, f'{name}.xlsx'))


if __name__ == '__main__':
    drivers_df = load_drivers_data()
    salary_df = load_salary_data()
    trips_df = load_trips_data(recalculate=False)
    summary_df = compute_summary_df(drivers_df, salary_df, trips_df)
