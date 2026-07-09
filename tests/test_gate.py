"""Tests for the promotion gate (walk-forward CV + interval-based promotion)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cto.models.gate import (
    auprc_logit_ci,
    evaluate,
    nadeau_bengio_ttest,
    promotion_decision_single,
    promotion_decision_walkforward,
    walk_forward_folds,
)


def _synth(n_per_year=200, years=(2019, 2020, 2021, 2022, 2023, 2024), pos_rate=0.2, seed=0):
    """Synthetic (completion_date, y) spanning several years."""
    rng = np.random.default_rng(seed)
    dates, y = [], []
    for yr in years:
        for _ in range(n_per_year):
            m, d = rng.integers(1, 13), rng.integers(1, 28)
            dates.append(pd.Timestamp(f"{yr}-{m:02d}-{d:02d}"))
            y.append(int(rng.random() < pos_rate))
    return pd.Series(pd.to_datetime(dates)), np.array(y)


def _good_probs(y_true, rng):
    """Well-separated predictions (positives score high)."""
    return np.where(
        y_true == 1, rng.uniform(0.5, 1.0, len(y_true)), rng.uniform(0.0, 0.5, len(y_true))
    )


# ── walk-forward folds ────────────────────────────────────────────────────────


def test_walk_forward_temporal_order():
    dates, y = _synth()
    folds = walk_forward_folds(dates, y, test_years=(2021, 2022, 2023, 2024))
    assert len(folds) == 4
    for f in folds:
        train_max = dates.iloc[f["train_idx"]].max()
        test_min = dates.iloc[f["test_idx"]].min()
        assert train_max < test_min, (
            f"fold {f['test_year']}: train max {train_max} not strictly before "
            f"test min {test_min}"
        )


def test_walk_forward_positive_per_fold():
    dates, y = _synth()
    folds = walk_forward_folds(dates, y)
    for f in folds:
        assert f["n_pos_test"] > 0


def test_walk_forward_raises_on_zero_positive_fold():
    dates, y = _synth()
    y = y.copy()
    y[(dates.dt.year == 2023).values] = 0  # wipe positives in one test window
    with pytest.raises(ValueError, match="positive"):
        walk_forward_folds(dates, y)


# ── AUPRC logit confidence interval (Boyd et al. 2013) ─────────────────────────


def test_auprc_logit_ci_in_bounds_and_wide_for_small_n():
    lo, hi = auprc_logit_ci(0.358, n_pos=20)
    assert 0.0 < lo < 0.358 < hi < 1.0
    assert (hi - lo) > 0.2  # n_pos=20 → wide interval (~±0.2)


def test_auprc_logit_ci_narrows_with_more_positives():
    w_small = auprc_logit_ci(0.5, n_pos=20)
    w_large = auprc_logit_ci(0.5, n_pos=1000)
    assert (w_large[1] - w_large[0]) < (w_small[1] - w_small[0])


# ── Nadeau-Bengio corrected resampled t-test ───────────────────────────────────


def test_nadeau_bengio_runs():
    diffs = np.array([0.02, -0.01, 0.03, 0.00])
    t, p = nadeau_bengio_ttest(diffs, n_train=800, n_test=200)
    assert np.isfinite(t)
    assert 0.0 <= p <= 1.0


# ── promotion decisions reject a worse challenger ──────────────────────────────


def test_worse_challenger_rejected_single():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    champ = _good_probs(y, rng)
    chall = rng.uniform(0, 1, 500)  # random → worse
    dec = promotion_decision_single(y, champ, chall)
    assert dec["promote"] is False


def test_worse_challenger_rejected_walkforward():
    dates, y = _synth(seed=2)
    folds = walk_forward_folds(dates, y)
    rng = np.random.default_rng(3)
    y_by_fold, champ_by_fold, chall_by_fold = [], [], []
    for f in folds:
        yt = y[f["test_idx"]]
        y_by_fold.append(yt)
        champ_by_fold.append(_good_probs(yt, rng))
        chall_by_fold.append(rng.uniform(0, 1, len(yt)))  # random → worse
    n_train = [len(f["train_idx"]) for f in folds]
    n_test = [len(f["test_idx"]) for f in folds]
    dec = promotion_decision_walkforward(y_by_fold, champ_by_fold, chall_by_fold, n_train, n_test)
    assert dec["promote"] is False


def test_clearly_better_challenger_promoted_single():
    """Gate must discriminate — a strongly-separated challenger is promoted."""
    rng = np.random.default_rng(7)
    n = 2000
    y = rng.integers(0, 2, n)
    champ = np.clip(y * 0.25 + rng.normal(0.4, 0.25, n), 0, 1)  # mediocre
    chall = np.clip(y * 0.65 + rng.normal(0.2, 0.15, n), 0, 1)  # strong
    dec = promotion_decision_single(y, champ, chall)
    assert dec["promote"] is True


def test_evaluate_returns_both_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, p)
    assert set(m) >= {"prauc", "auroc"}
    assert m["auroc"] == pytest.approx(1.0)
