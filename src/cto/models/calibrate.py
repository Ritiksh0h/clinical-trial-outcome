"""Isotonic calibration for XGBoost probability outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _IsotonicWrapper:
    """Thin wrapper that applies isotonic calibration to an XGBoost model.

    sklearn 1.9 removed cv="prefit" from CalibratedClassifierCV.
    This replicates the same behaviour without the deprecated parameter.
    """

    def __init__(self, model):
        self.model = model
        self._iso = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _IsotonicWrapper:
        from sklearn.isotonic import IsotonicRegression

        raw = self.model.predict_proba(X)[:, 1]
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        cal = self._iso.transform(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])

    def get_params(self, deep: bool = True) -> dict:  # noqa: ARG002
        return {}

    def set_params(self, **params) -> _IsotonicWrapper:  # noqa: ANN003
        return self


class _PlattWrapper:
    """Platt (sigmoid) calibration fit on a held-out val set.

    Small-n safe (Phase I/II gold val < 1000 points, where isotonic overfits). Fits a
    1-D logistic regression on the base model's raw positive-class probabilities → y_val.
    Monotonic, so PR-AUC / AUROC (rank metrics) are unchanged; only Brier/ECE improve.
    """

    def __init__(self, model):
        self.model = model
        self._lr = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _PlattWrapper:
        from sklearn.linear_model import LogisticRegression

        raw = self.model.predict_proba(X)[:, 1].reshape(-1, 1)
        self._lr = LogisticRegression(C=1e6, solver="lbfgs")  # near-unregularized Platt
        self._lr.fit(raw, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self._lr.predict_proba(raw)[:, 1]
        return np.column_stack([1 - cal, cal])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])


def calibrate_model(model, X_val: pd.DataFrame, y_val: np.ndarray, method: str = "isotonic"):
    """Fit calibration on the val set using the pre-fitted base model.

    method="sigmoid" → Platt (small-n safe, use for gold Phase I/II/III);
    method="isotonic" → isotonic (needs ≥1000 calibration points).
    """
    wrapper = _PlattWrapper(model) if method == "sigmoid" else _IsotonicWrapper(model)
    wrapper.fit(X_val, y_val)
    return wrapper


def plot_calibration(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    phase: int,
    output_dir: Path,
) -> None:
    """Save reliability curve + probability histogram for raw vs calibrated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability curve
    ax = axes[0]
    for probs, label, color in [
        (y_prob_raw, "Raw XGBoost", "#4C72B0"),
        (y_prob_cal, "Calibrated", "#DD8452"),
    ]:
        try:
            frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
            ax.plot(mean_pred, frac_pos, "s-", label=label, color=color)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_title(f"Phase {phase} Calibration Curve")
    ax.legend()

    # Probability histogram
    ax = axes[1]
    ax.hist(y_prob_raw, bins=40, alpha=0.6, label="Raw", color="#4C72B0")
    ax.hist(y_prob_cal, bins=40, alpha=0.6, label="Calibrated", color="#DD8452")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Phase {phase} Probability Distribution")
    ax.legend()

    fig.suptitle(f"Phase {phase} Calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"calibration_phase{phase}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration_phase%d.png", phase)
