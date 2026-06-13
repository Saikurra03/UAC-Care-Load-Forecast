from __future__ import annotations

import pandas as pd


def build_summary_report(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """Create a tabular summary report for export."""
    summary = df.describe(include="all").transpose()
    summary["title"] = title
    return summary.reset_index().rename(columns={"index": "feature"})


def export_report_to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
