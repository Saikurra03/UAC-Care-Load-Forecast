from __future__ import annotations

import pandas as pd

from .config import TARGET_RISK_BANDS


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Compute a normalized risk score for capacity and pressure forecasts."""
    if "Capacity Utilization" not in df.columns or "Net Pressure" not in df.columns:
        return pd.Series(index=df.index, data=0.0)
    capacity_norm = df["Capacity Utilization"].clip(0, 1)
    pressure_norm = (df["Net Pressure"] - df["Net Pressure"].min()) / max(1.0, df["Net Pressure"].max() - df["Net Pressure"].min())
    return ((capacity_norm + pressure_norm) / 2).fillna(0.0)


def risk_category(score: float) -> str:
    """Assign a risk category label based on a normalized score."""
    for label, (lower, upper) in TARGET_RISK_BANDS.items():
        if lower <= score < upper:
            return label.capitalize()
    return "Critical"


def build_risk_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """Build a capacity risk dashboard from forecasted values."""
    risk_score = compute_risk_score(df)
    dashboard = df.copy()
    dashboard["risk_score"] = risk_score
    dashboard["risk_category"] = dashboard["risk_score"].apply(risk_category)
    return dashboard
