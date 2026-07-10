import numpy as np
import pytest
from cto.models.train import compute_metrics


def test_compute_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    m = compute_metrics(y, p)
    assert m["prauc"] == pytest.approx(1.0)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(0.0)


def test_compute_metrics_random():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 100)
    p = rng.uniform(0, 1, 100)
    m = compute_metrics(y, p)
    assert 0.3 < m["auroc"] < 0.7
    assert 0.0 <= m["ece"] <= 1.0


def test_compute_metrics_returns_all_keys():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    m = compute_metrics(y, p)
    for key in ["prauc", "auroc", "f1", "brier", "ece"]:
        assert key in m, f"Missing key: {key}"


class _StubModel:
    """Predicts a fixed positive-class probability array (order = rows of X)."""

    def __init__(self, probs):
        self._probs = np.asarray(probs)

    def predict_proba(self, X):
        p = self._probs[: len(X)]
        return np.column_stack([1 - p, p])


def test_platt_wrapper_calibrates_and_preserves_ranking():
    """Platt calibration keeps outputs in [0,1], is monotonic (AUROC unchanged), and
    improves Brier on a miscalibrated (over-confident) base model."""
    from cto.models.calibrate import calibrate_model
    from sklearn.metrics import brier_score_loss, roc_auc_score

    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    # over-confident raw scores: correct ranking but pushed toward 0/1
    raw = np.clip(np.where(y == 1, 0.95, 0.05) + rng.normal(0, 0.03, n), 0.001, 0.999)
    import pandas as pd

    X = pd.DataFrame({"f": np.arange(n)})
    cal = calibrate_model(_StubModel(raw), X, y, method="sigmoid")
    cal_p = cal.predict_proba(X)[:, 1]
    assert cal_p.min() >= 0.0 and cal_p.max() <= 1.0
    # monotonic → ranking (AUROC) unchanged
    assert roc_auc_score(y, cal_p) == pytest.approx(roc_auc_score(y, raw), abs=1e-9)
    # calibration should not worsen Brier here
    assert brier_score_loss(y, cal_p) <= brier_score_loss(y, raw) + 1e-6
