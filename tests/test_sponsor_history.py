"""Tests for sponsor track-record features — TDD, written before sponsor_history.py.

The temporal self-join is the highest-leakage-risk operation in the project: prior-trial
counts must include ONLY trials registered strictly before the current trial.
"""

import pandas as pd

from cto.features.sponsor_history import compute_sponsor_history


def make_sponsor_df():
    """Three sponsors: big_pharma (many trials), startup (few), unknown (1)."""
    return pd.DataFrame(
        {
            "nct_id": [f"NCT{i:07d}" for i in range(8)],
            "source": [
                "BIG_PHARMA",
                "BIG_PHARMA",
                "BIG_PHARMA",
                "BIG_PHARMA",  # 4 trials
                "STARTUP",
                "STARTUP",  # 2 trials
                "UNKNOWN_SPONSOR",  # 1 trial
                "BIG_PHARMA",  # 5th trial
            ],
            "study_first_posted_date": pd.to_datetime(
                [
                    "2015-01-01",
                    "2016-01-01",
                    "2017-01-01",
                    "2018-01-01",
                    "2015-06-01",
                    "2017-06-01",
                    "2016-03-01",
                    "2019-01-01",
                ]
            ),
            # completion ~6 months after posting — every prior completes before the next
            # same-sponsor trial registers, so outcome-known-before holds for all priors here.
            "completion_date": pd.to_datetime(
                [
                    "2015-07-01",
                    "2016-07-01",
                    "2017-07-01",
                    "2018-07-01",
                    "2015-12-01",
                    "2017-12-01",
                    "2016-09-01",
                    "2019-07-01",
                ]
            ),
            "overall_status": [
                "COMPLETED",
                "COMPLETED",
                "TERMINATED",
                "COMPLETED",
                "COMPLETED",
                "TERMINATED",
                "COMPLETED",
                "COMPLETED",
            ],
            "phase": ["PHASE1"] * 8,
        }
    )


def test_no_future_leakage():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    first_row = result[result["nct_id"] == "NCT0000000"].iloc[0]
    assert first_row["sponsor_prior_trial_count"] == 0


def test_prior_counts_increase_over_time():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    bp_rows = result[
        result["nct_id"].isin(
            ["NCT0000000", "NCT0000001", "NCT0000002", "NCT0000003", "NCT0000007"]
        )
    ].sort_values("nct_id")
    counts = bp_rows["sponsor_prior_trial_count"].values
    assert counts[0] == 0  # first trial: no prior
    assert counts[1] == 1  # second: 1 prior
    assert counts[4] == 4  # fifth: 4 prior


def test_unknown_sponsor_gets_zero_history():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    unk = result[result["nct_id"] == "NCT0000006"].iloc[0]
    assert unk["sponsor_prior_trial_count"] == 0
    assert unk["sponsor_is_established"] == 0


def test_completion_rate_computed_correctly():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    # NCT0000003 (4th BIG_PHARMA, 2018): priors NCT0(COMP), NCT1(COMP), NCT2(TERM) → 2/3
    row = result[result["nct_id"] == "NCT0000003"].iloc[0]
    assert abs(row["sponsor_prior_completion_rate"] - (2 / 3)) < 0.01


def test_is_established_threshold():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    # NCT0000007 is the 5th BIG_PHARMA trial — prior count = 4, not yet established
    row_4_prior = result[result["nct_id"] == "NCT0000007"].iloc[0]
    assert row_4_prior["sponsor_prior_trial_count"] == 4
    assert row_4_prior["sponsor_is_established"] == 0  # threshold is 5


def test_output_columns_present():
    df = make_sponsor_df()
    result = compute_sponsor_history(df)
    expected = {
        "nct_id",
        "sponsor_prior_trial_count",
        "sponsor_prior_phase_count",
        "sponsor_prior_completion_rate",
        "sponsor_prior_same_phase_completion_rate",
        "sponsor_is_established",
        "sponsor_is_large",
    }
    assert expected.issubset(set(result.columns))


def test_no_nct_id_in_leakage_blocklist():
    from cto.features.leakage import LEAKAGE_BLOCKLIST

    sponsor_cols = {
        "sponsor_prior_trial_count",
        "sponsor_prior_phase_count",
        "sponsor_prior_completion_rate",
        "sponsor_prior_same_phase_completion_rate",
        "sponsor_is_established",
        "sponsor_is_large",
    }
    assert not sponsor_cols.intersection(LEAKAGE_BLOCKLIST)


def test_same_date_not_counted_as_prior():
    """Strict-before with ties: two same-sponsor trials on the SAME date must not
    count each other. This is the subtle correctness point of the temporal join."""
    df = pd.DataFrame(
        {
            "nct_id": ["A", "B", "C"],
            "source": ["S", "S", "S"],
            "study_first_posted_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-01-01"]),
            "overall_status": ["COMPLETED", "COMPLETED", "COMPLETED"],
            "phase": ["PHASE2"] * 3,
        }
    )
    r = compute_sponsor_history(df).set_index("nct_id")["sponsor_prior_trial_count"]
    assert r["A"] == 0  # same date as B → not counted
    assert r["B"] == 0  # same date as A → not counted
    assert r["C"] == 2  # strictly after both → 2 priors


def test_one_prior_completion_rate_is_extreme():
    """A sponsor with exactly 1 prior has completion rate 0.0 or 1.0 — thin signal.
    Verify the value is computed (not filled) so the sparsity is visible, not hidden."""
    df = make_sponsor_df()
    result = compute_sponsor_history(df).set_index("nct_id")
    # NCT0000001 (2nd BIG_PHARMA, 2016): 1 prior (NCT0 COMPLETED, done 2015-07) → rate 1.0
    assert result.loc["NCT0000001", "sponsor_prior_trial_count"] == 1
    assert result.loc["NCT0000001", "sponsor_prior_completion_rate"] == 1.0


def test_registered_before_but_completed_after_excluded_from_rate():
    """ITEM 3 fix: a prior registered before but COMPLETED after the current trial's
    registration is COUNTED (registration-order) but EXCLUDED from the rate (outcome
    not knowable at registration)."""
    df = pd.DataFrame(
        {
            "nct_id": ["A", "B", "C"],
            "source": ["S", "S", "S"],
            "study_first_posted_date": pd.to_datetime(["2014-01-01", "2015-01-01", "2018-01-01"]),
            "completion_date": pd.to_datetime(["2016-01-01", "2020-01-01", "2022-01-01"]),
            "overall_status": ["COMPLETED", "TERMINATED", "COMPLETED"],
            "phase": ["PHASE2"] * 3,
        }
    )
    r = compute_sponsor_history(df).set_index("nct_id")
    # C registered 2018: A (comp 2016 < 2018) known+COMPLETED; B (comp 2020 >= 2018) excluded.
    assert r.loc["C", "sponsor_prior_trial_count"] == 2  # both A,B registered before → counted
    assert r.loc["C", "sponsor_prior_completion_rate"] == 1.0  # only A known → 1/1; B excluded
    # Old leaked code would give 0.5 here (A COMPLETED + B TERMINATED) / 2.


def test_null_completion_date_excluded_from_rate():
    """A prior that never reached a terminal state (completion_date NULL) is counted but
    excluded from the rate — its outcome was not knowable."""
    df = pd.DataFrame(
        {
            "nct_id": ["A", "B", "C"],
            "source": ["S", "S", "S"],
            "study_first_posted_date": pd.to_datetime(["2013-01-01", "2014-01-01", "2018-01-01"]),
            "completion_date": pd.to_datetime(["2015-01-01", None, "2022-01-01"]),
            "overall_status": ["COMPLETED", "RECRUITING", "COMPLETED"],
            "phase": ["PHASE2"] * 3,
        }
    )
    r = compute_sponsor_history(df).set_index("nct_id")
    # C registered 2018: A (comp 2015) known+COMPLETED; B (comp NaT) excluded from rate.
    assert r.loc["C", "sponsor_prior_trial_count"] == 2  # A,B both registered before
    assert r.loc["C", "sponsor_prior_completion_rate"] == 1.0  # only A known; B (NaT) excluded
