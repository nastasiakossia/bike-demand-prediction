import numpy as np
import pandas as pd

SEASON_MAP = {
        0: "spring",
        1: "summer",
        2: "fall",
        3: "winter"
    }

def load_data(path):
    """Load the bike-sharing dataset from a CSV file."""
    return pd.read_csv(path)

def add_datetime_parts(df):
    """Convert timestamp to datetime and extract calendar features."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    return df

def add_cyclical_time_features(df):
    """Add cyclical encoding for hour and season."""
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
    return df


def add_domain_features(df):
    """Add domain-specific engineered features.
    Encoding:
        1 -> regular weekday
        2 -> weekend only
        3 -> holiday only
        4 -> holiday and weekend
    """
    df = df.copy()
    df['season_name'] = df['season'].map(SEASON_MAP)
    df['is_weekend_holiday'] = (
            df['is_holiday'].astype(int) * 2 +
            df['is_weekend'].astype(int)
    )
    df['t_diff'] = df['t2'] - df['t1']
    return df


def add_derived_features(df):
    """Run the full feature-engineering pipeline."""
    df = add_datetime_parts(df)
    df = add_cyclical_time_features(df)
    df = add_domain_features(df)
    return df

def add_demand_category(df):
    """Bucket bike demand into low, medium, and high classes."""
    df = df.copy()
    cnt = df['cnt'].values
    q1 = np.quantile(cnt, 0.33)
    q2 = np.quantile(cnt, 0.66)

    def categorize(cnt_value):
        if cnt_value <= q1:
            return 0
        elif cnt_value <= q2:
            return 1
        else:
            return 2

    df['demand_class'] = df['cnt'].apply(categorize)
    return df

def analyze_data(df):
    """
    Print descriptive statistics, correlations, and seasonal temperature-difference summaries.
    """
    cols_to_exclude = ['hour_sin', 'hour_cos', 'season_sin', 'season_cos']

    df_for_analysis = df.drop(columns=cols_to_exclude)

    print("Descriptive statistics:")
    print(df_for_analysis.describe().to_string())
    print()
    print("Correlation matrix:")
    corr = df_for_analysis.select_dtypes(include='number').corr()
    print(corr.to_string())
    print()

    # Store the absolute correlation value for each unique column pair.
    corr_dict = {}
    for i in range(0, len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            col1 = corr.columns[i]
            col2 = corr.columns[j]
            corr_dict[(col1, col2)] = abs(corr.iloc[i, j])

    # Top 5 highest and lowest correlated pairs
    highest_corr = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    lowest_corr = sorted(corr_dict.items(), key=lambda x: x[1])[:5]

    print("Top 5 strongest absolute correlations: ")
    for index, (pair, value) in enumerate(highest_corr):
        print(f"{index + 1}. {pair} with {value:.6f}")
    print()
    print("Top 5 weakest absolute correlations: ")
    for index, (pair, value) in enumerate(lowest_corr):
        print(f"{index + 1}. {pair} with {value:.6f}")

    grouped = df.groupby('season_name')['t_diff'].mean()
    all_avg = df['t_diff'].mean()

    print("Average t_diff by season:")
    for season in ["spring", "summer", "fall", "winter"]:
        value = grouped.get(season, float('nan'))
        print(f"{season}: {value:.2f}")
    print(f"overall: {all_avg:.2f}")
