"""SHAP feature importance for trained XGBoost models."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SAMPLE_SIZE = 2000


def compute_shap(model, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values on a random sample of X (max 2000 rows)."""
    import shap

    sample = X.sample(min(_SAMPLE_SIZE, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(sample), sample


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    phase: int,
    output_dir: Path,
) -> None:
    """Save beeswarm + bar SHAP plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)

    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f"Phase {phase} SHAP Summary")
    plt.tight_layout()
    fig.savefig(output_dir / f"shap_phase{phase}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Bar (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(f"Phase {phase} Mean |SHAP|")
    plt.tight_layout()
    fig.savefig(output_dir / f"shap_bar_phase{phase}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP plots for phase %d", phase)


def get_top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    n: int = 10,
) -> list[dict]:
    """Return top N features by mean |SHAP|."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:n]
    return [{"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])} for i in idx]
