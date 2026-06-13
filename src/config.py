from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"

SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".parquet"]
FORECAST_HORIZONS = [7, 14, 30, 60, 90]
DEFAULT_FORECAST_HORIZON = 30
DEFAULT_FREQ = "D"
TARGET_RISK_BANDS = {
    "low": (0.0, 0.5),
    "medium": (0.5, 0.75),
    "high": (0.75, 0.9),
    "critical": (0.9, 1.0),
}

CANDIDATE_MODELS = [
    "Naive", "SeasonalNaive", "MovingAverage", "HoltWinters", "ARIMA", "AutoARIMA",
    "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost"
]
