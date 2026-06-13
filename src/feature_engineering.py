from __future__ import annotations

import pandas as pd

LAG_WINDOWS = [1, 3, 7, 14, 30, 60]
ROLL_WINDOWS = [7, 14, 30, 60]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based time features from the index."""
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["is_weekend"] = df.index.dayofweek >= 5
    df["is_month_start"] = df.index.is_month_start
    df["is_month_end"] = df.index.is_month_end
    return df


def add_lag_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add lag features for specified numeric columns."""
    df = df.copy()
    for column in columns:
        for lag in LAG_WINDOWS:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add rolling statistics for specified numeric columns."""
    df = df.copy()
    for column in columns:
        for window in ROLL_WINDOWS:
            df[f"{column}_rolling_mean_{window}"] = df[column].shift(1).rolling(window=window, min_periods=1).mean()
            df[f"{column}_rolling_median_{window}"] = df[column].shift(1).rolling(window=window, min_periods=1).median()
            df[f"{column}_rolling_std_{window}"] = df[column].shift(1).rolling(window=window, min_periods=1).std()
            df[f"{column}_rolling_min_{window}"] = df[column].shift(1).rolling(window=window, min_periods=1).min()
            df[f"{column}_rolling_max_{window}"] = df[column].shift(1).rolling(window=window, min_periods=1).max()
    return df


def add_growth_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add growth and momentum features for numeric columns."""
    df = df.copy()
    for column in columns:
        df[f"{column}_pct_change_1"] = df[column].pct_change(periods=1)
        df[f"{column}_momentum_7"] = df[column].diff(periods=7)
        df[f"{column}_trend_strength_14"] = df[column].rolling(window=14, min_periods=1).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix with calendar, lag, rolling, and growth features."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df = add_calendar_features(df)
    df = add_lag_features(df, numeric_cols)
    df = add_rolling_features(df, numeric_cols)
    df = add_growth_features(df, numeric_cols)
    df = df.dropna().copy()
    return df
