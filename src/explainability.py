from __future__ import annotations

import pandas as pd


import numpy as np


def summarize_feature_importance(model: object, feature_names: list[str]) -> pd.DataFrame:
    """Return a feature importance summary when available."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = getattr(model, "coef_")
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(by="importance", ascending=False)


def extract_shap_insights(model: object, X: pd.DataFrame) -> pd.DataFrame:
    """Provide a placeholder for SHAP explanations if supported."""
    try:
        import shap
    except ImportError:
        return pd.DataFrame(columns=["feature", "shap_value"])
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    mean_abs = pd.DataFrame({"feature": X.columns, "shap_value": np.abs(shap_values.values).mean(axis=0)})
    return mean_abs.sort_values(by="shap_value", ascending=False)
