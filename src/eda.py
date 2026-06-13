from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)


def build_data_quality_report(df: pd.DataFrame) -> dict[str, object]:
    """Generate a data quality report for the dataset."""
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    report = {
        "shape": df.shape,
        "null_rate": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "cardinality": df.nunique(dropna=False).to_dict(),
        "numeric_columns": numeric_columns,
        "date_range": [str(df.index.min()), str(df.index.max())] if not df.index.empty else [None, None],
        "missing_date_count": int(df.index.isna().sum()),
        "summary_statistics": df.describe(include="all").to_dict(),
    }
    return report


def save_data_quality_report(report: dict[str, object]) -> Path:
    """Save the data quality report as JSON and CSV artifacts."""
    artifact_path = Path(ARTIFACTS_DIR) / "data_quality_report.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("Saved data quality report to %s", artifact_path)
    return artifact_path


def save_eda_summary(df: pd.DataFrame) -> Path:
    """Save a summary table for exploratory data analysis."""
    path = Path(REPORTS_DIR) / "eda_summary.csv"
    df.describe(include="all").to_csv(path)
    logger.info("Saved EDA summary to %s", path)
    return path
