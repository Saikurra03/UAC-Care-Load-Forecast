from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import BASE_DIR, DATA_DIR, SUPPORTED_EXTENSIONS
from .utils import sanitize_numeric_series

logger = logging.getLogger(__name__)


def discover_dataset_files() -> list[Path]:
    """Discover supported dataset files in root and data directories."""
    files: list[Path] = []
    search_dirs = [BASE_DIR, DATA_DIR]
    for folder in search_dirs:
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
    files.sort(key=lambda p: p.name)
    logger.info("Discovered %s supported dataset file(s).", len(files))
    return files


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    except Exception:
        return pd.read_csv(path, dtype=str, engine="python", encoding="utf-8-sig", keep_default_na=False)


def _read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, dtype=str)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset from a supported file path into a DataFrame."""
    logger.info("Loading dataset from %s", path)
    if path.suffix.lower() == ".csv":
        df = _read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = _read_excel(path)
    elif path.suffix.lower() == ".parquet":
        df = _read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    df.columns = [str(column).strip() for column in df.columns]
    return df


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer simplified column types from a DataFrame."""
    column_types: dict[str, str] = {}
    for column in df.columns:
        sample = df[column].dropna().astype(str).head(50).tolist()
        if all(_is_datetime_like(value) for value in sample if str(value).strip()):
            column_types[column] = "datetime"
            continue
        numeric_values = sanitize_numeric_series(sample)
        if sum(1 for value in numeric_values if value is not None) >= len(sample) * 0.6:
            column_types[column] = "numeric"
        else:
            column_types[column] = "text"
    return column_types


def _is_datetime_like(value: object) -> bool:
    try:
        pd.to_datetime(value, infer_datetime_format=True, errors="raise")
        return True
    except Exception:
        return False


def detect_date_column(df: pd.DataFrame) -> str | None:
    """Detect the primary date column in a DataFrame."""
    candidates: dict[str, float] = {}
    for column in df.columns:
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        parsed = pd.to_datetime(values, infer_datetime_format=True, errors="coerce")
        score = parsed.notna().mean()
        candidates[column] = score
    if not candidates:
        return None
    best = max(candidates.items(), key=lambda item: item[1])
    logger.info("Date column candidates: %s", candidates)
    if best[1] < 0.6:
        return None
    return best[0]


def build_schema_report(df: pd.DataFrame) -> dict[str, Any]:
    """Build a data report describing the schema and quality of a dataset."""
    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "null_percentages": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "cardinality": df.nunique(dropna=False).to_dict(),
    }
    return report
