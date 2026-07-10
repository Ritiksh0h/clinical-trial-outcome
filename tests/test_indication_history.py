"""Tests for indication (therapeutic-area) history features — TDD.

Same leak profile as sponsor history: RATE features must use outcome-known-before,
COUNT features use registration-order. Written before indication_history.py.
"""

import pandas as pd
from cto.features.indication_history import compute_indication_history


def _conditions(mapping: dict[str, str]) -> pd.DataFrame:
    """mapping: nct_id -> mesh term (lowercased text)."""
    return pd.DataFrame({"nct_id": list(mapping), "downcase_mesh_term": list(mapping.values())})


def test_other_bucket_assignment():
    """A cancer condition → ONCOLOGY; an unmatched condition → OTHER."""
    studies = pd.DataFrame(
        {
            "nct_id": ["A", "B"],
            "study_first_posted_date": pd.to_datetime(["2015-01-01", "2015-01-01"]),
            "completion_date": pd.to_datetime(["2016-01-01", "2016-01-01"]),
            "overall_status": ["COMPLETED", "COMPLETED"],
            "phase": ["PHASE2", "PHASE2"],
        }
    )
    cond = _conditions({"A": "lung cancer", "B": "left-handedness"})
    r = compute_indication_history(studies, cond).set_index("nct_id")
    # ta_bucket: ONCOLOGY != OTHER; OTHER is code 0
    assert r.loc["A", "ta_bucket"] != 0  # oncology, mapped
    assert r.loc["B", "ta_bucket"] == 0  # unmatched → OTHER


def test_counts_registration_order():
    """Same-TA prior counts use registration order (strictly earlier)."""
    studies = pd.DataFrame(
        {
            "nct_id": ["A", "B", "C"],
            "study_first_posted_date": pd.to_datetime(["2015-01-01", "2016-01-01", "2017-01-01"]),
            "completion_date": pd.to_datetime(["2015-07-01", "2016-07-01", "2017-07-01"]),
            "overall_status": ["COMPLETED", "COMPLETED", "COMPLETED"],
            "phase": ["PHASE2"] * 3,
        }
    )
    cond = _conditions({"A": "breast cancer", "B": "colon cancer", "C": "lung cancer"})
    r = compute_indication_history(studies, cond).set_index("nct_id")
    assert r.loc["A", "ta_prior_trial_count"] == 0
    assert r.loc["B", "ta_prior_trial_count"] == 1
    assert r.loc["C", "ta_prior_trial_count"] == 2


def test_registered_before_but_completed_after_excluded_from_rate():
    """IDENTICAL to the sponsor Item 3 test: a same-TA prior registered before but
    completed after the current registration is COUNTED but EXCLUDED from the rate."""
    studies = pd.DataFrame(
        {
            "nct_id": ["A", "B", "C"],
            "study_first_posted_date": pd.to_datetime(["2014-01-01", "2015-01-01", "2018-01-01"]),
            "completion_date": pd.to_datetime(["2016-01-01", "2020-01-01", "2022-01-01"]),
            "overall_status": ["COMPLETED", "TERMINATED", "COMPLETED"],
            "phase": ["PHASE2"] * 3,
        }
    )
    cond = _conditions({"A": "melanoma", "B": "glioma", "C": "sarcoma"})  # all ONCOLOGY
    r = compute_indication_history(studies, cond).set_index("nct_id")
    assert r.loc["C", "ta_prior_trial_count"] == 2  # A,B both registered before
    assert r.loc["C", "ta_prior_completion_rate"] == 1.0  # only A known (COMPLETED); B excluded
    # Leaked version would give 0.5 (A COMPLETED + B TERMINATED)/2.


def test_no_history_gets_base_rate_fill():
    """First trial in a TA has count 0 and a filled (finite, non-NaN) rate."""
    studies = pd.DataFrame(
        {
            "nct_id": ["A", "B"],
            "study_first_posted_date": pd.to_datetime(["2015-01-01", "2016-01-01"]),
            "completion_date": pd.to_datetime(["2015-07-01", "2016-07-01"]),
            "overall_status": ["COMPLETED", "COMPLETED"],
            "phase": ["PHASE2"] * 2,
        }
    )
    cond = _conditions({"A": "diabetes", "B": "hiv infection"})  # different TAs
    r = compute_indication_history(studies, cond).set_index("nct_id")
    assert r.loc["A", "ta_prior_trial_count"] == 0
    assert pd.notna(r.loc["A", "ta_prior_completion_rate"])  # filled, not NaN
    assert 0.0 <= r.loc["A", "ta_prior_completion_rate"] <= 1.0


def test_output_columns_present():
    studies = pd.DataFrame(
        {
            "nct_id": ["A"],
            "study_first_posted_date": pd.to_datetime(["2015-01-01"]),
            "completion_date": pd.to_datetime(["2016-01-01"]),
            "overall_status": ["COMPLETED"],
            "phase": ["PHASE2"],
        }
    )
    cond = _conditions({"A": "asthma"})
    r = compute_indication_history(studies, cond)
    expected = {
        "nct_id",
        "ta_prior_trial_count",
        "ta_prior_same_phase_count",
        "ta_prior_completion_rate",
        "ta_prior_same_phase_completion_rate",
        "ta_bucket",
    }
    assert expected.issubset(set(r.columns))


def test_no_ta_col_in_leakage_blocklist():
    from cto.features.leakage import LEAKAGE_BLOCKLIST

    ta_cols = {
        "ta_prior_trial_count",
        "ta_prior_same_phase_count",
        "ta_prior_completion_rate",
        "ta_prior_same_phase_completion_rate",
        "ta_bucket",
    }
    assert not ta_cols.intersection(LEAKAGE_BLOCKLIST)
