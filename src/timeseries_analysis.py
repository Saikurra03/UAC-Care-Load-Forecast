from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


def run_adf_test(series: pd.Series) -> dict[str, Any]:
    """Run Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
    }


def run_kpss_test(series: pd.Series) -> dict[str, Any]:
    """Run KPSS test for stationarity."""
    statistic, p_value, _, critical_values = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "kpss_statistic": statistic,
        "p_value": p_value,
        "critical_values": critical_values,
    }


def decompose_series(series: pd.Series, period: int | None = None) -> dict[str, pd.Series]:
    """Decompose a time series using STL."""
    if period is None:
        period = max(2, int(len(series) / 10))
    series = series.dropna()
    decomposition = STL(series, period=period, robust=True).fit()
    return {
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "resid": decomposition.resid,
    }


def summary_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Produce diagnostic summaries for numeric time series."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    diagnostics: dict[str, Any] = {}
    for column in numeric_cols:
        series = df[column].dropna()
        diagnostics[column] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "adf": run_adf_test(series),
            "kpss": run_kpss_test(series),
        }
    logger.info("Computed diagnostics for %s numeric columns.", len(numeric_cols))
    return diagnostics
