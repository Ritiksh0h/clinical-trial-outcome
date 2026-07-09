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
