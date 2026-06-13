from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None

try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from .config import MODELS_DIR
from .evaluation import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    name: str
    predictions: pd.Series
    metrics: dict[str, float]
    model: Any | None = None


def naive_forecast(series: pd.Series, horizon: int) -> pd.Series:
    return pd.Series([series.iloc[-1]] * horizon, index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"))


def moving_average_forecast(series: pd.Series, horizon: int, window: int = 7) -> pd.Series:
    value = series.tail(window).mean()
    return pd.Series([value] * horizon, index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"))


def holt_winters_forecast(series: pd.Series, horizon: int) -> pd.Series:
    if ExponentialSmoothing is None:
        raise ImportError("statsmodels is not installed")
    model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
    fitted = model.fit(optimized=True)
    future = fitted.forecast(horizon)
    return future


def auto_arima_forecast(series: pd.Series, horizon: int) -> pd.Series:
    if auto_arima is None:
        raise ImportError("pmdarima is not installed")
    model = auto_arima(series, seasonal=False, error_action="ignore", suppress_warnings=True, stepwise=True)
    fitted = model.fit(series)
    forecast = fitted.predict(n_periods=horizon)
    return pd.Series(forecast, index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"))


def train_ml_models(X: pd.DataFrame, y: pd.Series, horizon: int) -> list[ForecastResult]:
    results: list[ForecastResult] = []
    X_train, y_train = X.iloc[:-horizon], y.iloc[:-horizon]
    X_pred = X.iloc[-horizon:]
    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    if LGBMRegressor is not None:
        candidates["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42)
    if CatBoostRegressor is not None:
        candidates["CatBoost"] = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            predictions = pd.Series(model.predict(X_pred), index=X_pred.index)
            metrics = compute_metrics(y.iloc[-horizon:], predictions)
            results.append(ForecastResult(name=name, predictions=predictions, metrics=metrics, model=model))
        except Exception as exc:
            logger.warning("Could not train %s: %s", name, exc)
    return results


def evaluate_statistical_models(series: pd.Series, horizon: int) -> list[ForecastResult]:
    results: list[ForecastResult] = []
    y_true = series.iloc[-horizon:]
    try:
        predictions = naive_forecast(series[:-horizon], horizon)
        results.append(ForecastResult(name="Naive", predictions=predictions, metrics=compute_metrics(y_true, predictions)))
    except Exception:
        pass
    try:
        predictions = moving_average_forecast(series[:-horizon], horizon)
        results.append(ForecastResult(name="MovingAverage", predictions=predictions, metrics=compute_metrics(y_true, predictions)))
    except Exception:
        pass
    try:
        predictions = holt_winters_forecast(series[:-horizon], horizon)
        results.append(ForecastResult(name="HoltWinters", predictions=predictions, metrics=compute_metrics(y_true, predictions)))
    except Exception as exc:
        logger.warning("HoltWinters failed: %s", exc)
    if auto_arima is not None:
        try:
            predictions = auto_arima_forecast(series[:-horizon], horizon)
            results.append(ForecastResult(name="AutoARIMA", predictions=predictions, metrics=compute_metrics(y_true, predictions)))
        except Exception as exc:
            logger.warning("AutoARIMA failed: %s", exc)
    return results


def select_best_forecast(results: list[ForecastResult]) -> ForecastResult | None:
    ranked = sorted(results, key=lambda result: result.metrics.get("mae", float("inf")))
    return ranked[0] if ranked else None


def save_model_artifact(model: Any, name: str) -> Path:
    import pickle

    path = Path(MODELS_DIR) / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)
    logger.info("Saved model artifact to %s", path)
    return path
