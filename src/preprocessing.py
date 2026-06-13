from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, DEFAULT_FREQ
from .data_loader import detect_date_column
from .utils import sanitize_numeric_series

logger = logging.getLogger(__name__)


def _convert_series_to_numeric(series: pd.Series) -> pd.Series:
    cleaned = sanitize_numeric_series(series.astype(object).tolist())
    return pd.Series([np.nan if value is None else value for value in cleaned], index=series.index)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the dataset with date parsing and numeric conversions."""
    df = df.copy()
    date_column = detect_date_column(df)
    if date_column is None:
        raise ValueError("Could not infer a date column from the dataset.")
    df[date_column] = pd.to_datetime(df[date_column], infer_datetime_format=True, errors="coerce")
    if df[date_column].isna().any():
        logger.warning("Date conversion produced missing values in %s", date_column)
    df = df.dropna(subset=[date_column]).drop_duplicates()
    for column in df.columns:
        if column == date_column:
            continue
        series = df[column].astype(str).replace({"": None, "na": None, "n/a": None, "null": None, "None": None})
        numeric = _convert_series_to_numeric(series)
        if numeric.notna().sum() >= len(series) * 0.4:
            df[column] = numeric
    df = df.sort_values(date_column).reset_index(drop=True)
    df = df.rename(columns={date_column: "date"})
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index, infer_datetime_format=True, errors="coerce")
    df = df[~df.index.duplicated(keep="first")]
    df = fill_missing_dates(df)
    df = impute_missing_values(df)
    cleaned_path = Path(ARTIFACTS_DIR) / "cleaned_dataset.csv"
    df.to_csv(cleaned_path)
    logger.info("Saved cleaned dataset to %s", cleaned_path)
    return df


def fill_missing_dates(df: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    """Insert missing date rows based on detected frequency."""
    if freq is None:
        freq = pd.infer_freq(df.index)
    if freq is None:
        freq = DEFAULT_FREQ
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using interpolation and forward/backward fill."""
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column] = df[column].interpolate(method="linear", limit_direction="both")
        df[column] = df[column].ffill().bfill()
    return df


def detect_target_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric columns that are valid targets for forecasting."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_columns


def derive_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived operational metrics used for capacity and pressure forecasting."""
    df = df.copy()
    if "Children transferred out of CBP custody" in df.columns and "Children discharged from HHS Care" in df.columns:
        df["Net Pressure"] = df["Children transferred out of CBP custody"] - df["Children discharged from HHS Care"]
        df["Pressure Ratio"] = df["Net Pressure"] / (df["Children discharged from HHS Care"].replace(0, np.nan).abs())
    if "Children in HHS Care" in df.columns:
        max_care = df["Children in HHS Care"].max() or 1
        df["Capacity Utilization"] = df["Children in HHS Care"] / max_care
    if "Children apprehended and placed in CBP custody*" in df.columns and "Children discharged from HHS Care" in df.columns:
        df["Intake vs Exit Imbalance"] = df["Children apprehended and placed in CBP custody*"] - df["Children discharged from HHS Care"]
    return df
