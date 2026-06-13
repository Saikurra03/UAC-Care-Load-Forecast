from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_time_series(df: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for column in columns:
        if column in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode="lines+markers", name=column))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value", legend_title="Series")
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    corr = df.select_dtypes(include=["number"]).corr()
    return px.imshow(corr, text_auto=True, title="Correlation Matrix")


def plot_forecast(series: pd.Series, forecast: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecast", mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value")
    return fig


def plot_risk_categories(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_category"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
    return px.bar(counts, x=counts.index, y=counts.values, title="Risk Category Breakdown", labels={"x": "Category", "y": "Count"})
