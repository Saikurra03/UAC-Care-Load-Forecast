from __future__ import annotations

import pandas as pd


def choose_best_model_by_metric(scores: dict[str, dict[str, float]], metric: str = "mae") -> str | None:
    """Choose the best model based on a selected evaluation metric."""
    candidates = [(name, values.get(metric, float("inf"))) for name, values in scores.items()]
    candidates = [item for item in candidates if item[1] is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]


def build_leaderboard(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a leaderboard from model metrics."""
    rows = []
    for name, values in metrics.items():
        row = {"model": name, **values}
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by="mae").reset_index(drop=True)
