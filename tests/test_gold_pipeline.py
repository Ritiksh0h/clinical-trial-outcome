"""Tests for the gold featurize pipeline + frozen gold test set (Phase 2 Step 4).

Data-dependent: skip until `python -m cto.pipelines.featurize_gold` has run.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

_PROCESSED = Path(__file__).parents[1] / "data" / "processed"
_FROZEN = _PROCESSED / "gold_test_nct_ids.json"

pytestmark = pytest.mark.skipif(
    not _FROZEN.exists(), reason="run `python -m cto.pipelines.featurize_gold` first"
)


def test_gold_matrix_has_history_and_no_leak_features():
    """Gold matrix must include the sponsor + indication history features and must NOT
    contain the dropped conduct-accrued soft-leak features."""
    cols = set(pd.read_parquet(_PROCESSED / "features_gold_phase1_train.parquet").columns)
    assert {
        "sponsor_prior_completion_rate",
        "sponsor_is_established",
        "ta_prior_completion_rate",
        "ta_bucket",
    } <= cols
    assert not ({"number_of_facilities", "num_countries", "is_multinational"} & cols)


def test_gold_positive_rates_match_expected():
    """Combined (train+val+test) per-phase gold positive rate matches the confirmed
    membership-routed distribution (Phase I ~0.19, II ~0.30, III ~0.50)."""
    expected = {1: (0.15, 0.25), 2: (0.25, 0.38), 3: (0.45, 0.62)}
    for phase, (lo, hi) in expected.items():
        ys = [
            pd.read_parquet(_PROCESSED / f"labels_gold_phase{phase}_{s}.parquet")["y"]
            for s in ("train", "val", "test")
        ]
        rate = pd.concat(ys).mean()
        assert lo <= rate <= hi, f"phase {phase} combined pos_rate {rate:.3f} outside [{lo},{hi}]"


def test_frozen_counts_match_test_matrices():
    """One-split-two-uses: each phase's frozen test-id count equals that phase's test
    matrix row count (both come from the same split object)."""
    frozen = json.loads(_FROZEN.read_text())
    for phase in (1, 2, 3):
        n_matrix = len(pd.read_parquet(_PROCESSED / f"features_gold_phase{phase}_test.parquet"))
        assert (
            len(frozen[f"phase{phase}"]) == n_matrix
        ), f"phase {phase}: frozen {len(frozen[f'phase{phase}'])} != test matrix {n_matrix}"


def test_frozen_structure_and_union():
    frozen = json.loads(_FROZEN.read_text())
    assert set(frozen) == {"phase1", "phase2", "phase3", "all"}
    union = set(frozen["phase1"]) | set(frozen["phase2"]) | set(frozen["phase3"])
    assert set(frozen["all"]) == union  # 'all' is exactly the dedup union


def test_no_leakage_in_gold_matrix():
    from cto.features.leakage import LEAKAGE_BLOCKLIST

    for phase in (1, 2, 3):
        cols = set(
            pd.read_parquet(_PROCESSED / f"features_gold_phase{phase}_train.parquet").columns
        )
        assert not (cols & LEAKAGE_BLOCKLIST)
