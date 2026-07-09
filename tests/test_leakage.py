"""
Leakage gate tests — must be green before any model work begins.
These are the most important tests in the project.
"""

import pandas as pd
import pytest
from cto.features.leakage import (
    LEAKAGE_BLOCKLIST,
    assert_no_leakage,
    drop_leakage_columns,
)

# ── Blocklist sanity ──────────────────────────────────────────────────────────


def test_blocklist_is_nonempty():
    assert len(LEAKAGE_BLOCKLIST) >= 20, "Blocklist suspiciously small"


@pytest.mark.parametrize(
    "col",
    [
        # CTO LF columns
        "why_stopped",
        "pvalues",
        "serious_ae",
        "death_ae",
        "all_ae",
        "results_reported",
        "results_first_posted_date",
        "patient_drop",
        "stock_price",
        "new_headlines",
        "num_patients",
        "amendments",
        "pred_proba",
        "hint_train",
        "hint_train2",
        "hint_train3",
        "status",
        "status2",
        "gpt",
        "gpt2",
        "linkage",
        "linkage2",
        "update_more_recent",
        "sites",
        # AACT post-hoc
        "actual_duration",
        "were_results_reported",
        "last_known_status",
        "fdaaa801_violation",
        "limitations_and_caveats",
        "overall_status",
        "last_update_posted_date",
    ],
)
def test_known_offenders_in_blocklist(col):
    assert col in LEAKAGE_BLOCKLIST, f"'{col}' must be in LEAKAGE_BLOCKLIST"


# ── assert_no_leakage ─────────────────────────────────────────────────────────


def test_clean_df_passes():
    df = pd.DataFrame({"phase": [1, 2], "enrollment_log": [3.5, 4.1]})
    assert_no_leakage(df)  # must not raise


def test_single_leaked_column_raises():
    df = pd.DataFrame({"phase": [1], "pvalues": [0.03]})
    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        assert_no_leakage(df)


def test_multiple_leaked_columns_all_reported():
    df = pd.DataFrame({"phase": [1], "pvalues": [0.03], "serious_ae": [2]})
    with pytest.raises(ValueError) as exc_info:
        assert_no_leakage(df)
    msg = str(exc_info.value)
    assert "pvalues" in msg
    assert "serious_ae" in msg


def test_context_string_appears_in_error():
    df = pd.DataFrame({"why_stopped": ["business"]})
    with pytest.raises(ValueError, match="build_phase1_features"):
        assert_no_leakage(df, context="build_phase1_features")


def test_nct_id_is_allowed():
    """nct_id is an identifier, not a feature — but it must not be blocked."""
    df = pd.DataFrame({"nct_id": ["NCT0001"], "phase": [1]})
    assert_no_leakage(df)  # must not raise


# ── drop_leakage_columns ──────────────────────────────────────────────────────


def test_drop_removes_blocked_columns():
    df = pd.DataFrame({"phase": [1], "pvalues": [0.03], "enrollment_log": [3.5]})
    cleaned = drop_leakage_columns(df, warn=False)
    assert "pvalues" not in cleaned.columns
    assert "phase" in cleaned.columns
    assert "enrollment_log" in cleaned.columns


def test_drop_on_clean_df_is_noop():
    df = pd.DataFrame({"phase": [1], "enrollment_log": [3.5]})
    cleaned = drop_leakage_columns(df, warn=False)
    assert list(cleaned.columns) == list(df.columns)


# ── CTO label-side isolation ──────────────────────────────────────────────────

CTO_LF_COLUMNS = [
    "hint_train",
    "hint_train2",
    "hint_train3",
    "status",
    "status2",
    "gpt",
    "gpt2",
    "linkage",
    "linkage2",
    "stock_price",
    "results_reported",
    "new_headlines",
    "pvalues",
    "update_more_recent",
    "sites",
    "serious_ae",
    "patient_drop",
    "num_patients",
    "death_ae",
    "amendments",
    "all_ae",
    "pred_proba",
]


def test_all_cto_lf_columns_in_blocklist():
    """Every CTO labeling-function column must be blocked from feature use."""
    missing = [c for c in CTO_LF_COLUMNS if c not in LEAKAGE_BLOCKLIST]
    assert not missing, (
        f"CTO LF columns not in blocklist: {missing}\n"
        f"These columns were used to *generate* the weak labels and "
        f"must never be used as features."
    )


def test_pred_proba_is_blocked():
    """pred_proba is CTO's output label proxy — must be blocked."""
    assert "pred_proba" in LEAKAGE_BLOCKLIST


def test_nct_id_not_blocked():
    """nct_id is the join key — must not be accidentally blocked."""
    assert "nct_id" not in LEAKAGE_BLOCKLIST


def test_estimated_enrollment_only():
    """enrollment_log must come from ESTIMATED (registration-time) rows only.

    ACTUAL enrollment is completion-time soft leakage and must yield NaN, never a value.
    """
    import numpy as np
    from cto.features.build import _compute_enrollment_log

    enrollment = pd.Series([100, 200, 300, 400])
    enrollment_type = pd.Series(["ESTIMATED", "ACTUAL", None, "estimated"])
    out = _compute_enrollment_log(enrollment, enrollment_type)

    assert not np.isnan(out[0]), "ESTIMATED row must produce a value"
    assert np.isnan(out[1]), "ACTUAL row must be NaN (soft leakage), not a value"
    assert np.isnan(out[2]), "unknown enrollment_type must be NaN"
    assert not np.isnan(out[3]), "case-insensitive 'estimated' must produce a value"
    assert out[0] == pytest.approx(np.log1p(100))


def test_raw_enrollment_not_hard_blocked():
    """The raw `enrollment` column must NOT be a hard blocklist entry — build.py
    legitimately consumes it (ESTIMATED-gated). Blocklisting the name would make
    drop_leakage_columns() delete it and re-break enrollment_log."""
    assert "enrollment" not in LEAKAGE_BLOCKLIST
