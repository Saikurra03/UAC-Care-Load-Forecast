from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from .config import ARTIFACTS_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR


def setup_logging(log_file: Path | str | None = None, level: int = logging.INFO) -> None:
    """Configure root logger and file output."""
    target = Path(log_file) if log_file else LOGS_DIR / "app.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(target, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def ensure_directories() -> None:
    """Create application directories for artifacts and outputs."""
    for path in [ARTIFACTS_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(value: object) -> float | None:
    """Convert a value to float safely, returning None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_number_like(value: object) -> bool:
    """Check whether a value can be interpreted as a number."""
    return safe_float(value) is not None


def sanitize_numeric_series(values: Iterable[object]) -> list[float | None]:
    """Clean numeric series values and preserve missing cases."""
    result: list[float | None] = []
    for raw in values:
        if raw is None:
            result.append(None)
            continue
        text = str(raw).strip().replace(",", "").replace("$", "")
        if text in {"", "na", "n/a", "none", "null", "nan"}:
            result.append(None)
            continue
        try:
            result.append(float(text))
        except ValueError:
            result.append(None)
    return result
