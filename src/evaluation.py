from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float((np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100)


def symmetric_mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return float((np.abs(y_true - y_pred)[mask] / denominator[mask]).mean() * 100)


def forecast_bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float((y_pred - y_true).mean())


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    r2 = float(np.nan if np.std(y_true) == 0 else 1 - np.sum(residual ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "bias": forecast_bias(y_true, y_pred),
    }


def leaderboard(results: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values(by="mae").reset_index(drop=True)
