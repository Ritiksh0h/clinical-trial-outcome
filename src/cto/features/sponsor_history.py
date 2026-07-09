"""
Sponsor track-record features (temporal, leakage-safe).

Two distinct temporal windows, because "registered before" and "outcome known before"
are NOT the same thing:

  COUNT features (sponsor_prior_trial_count, sponsor_prior_phase_count, and the
  is_established / is_large flags) use REGISTRATION order — a prior counts if
  other.study_first_posted_date < this.study_first_posted_date. Registration order is
  valid for counting: how many trials a sponsor has registered is known at registration.

  RATE features (sponsor_prior_completion_rate, sponsor_prior_same_phase_completion_rate)
  use OUTCOME-KNOWN order — a prior counts toward the rate only if its
  completion_date < this.study_first_posted_date. A prior registered earlier but still
  running (or completing later) had NO known outcome at the current trial's registration,
  so using its eventual status would leak the future (audit Item 3: ~37.6% of counted
  priors were leaked this way, distorting the rate ~8pp). Priors with null completion or
  completion on/after the current registration are EXCLUDED from the rate.

Both use an aggregate + cumulative approach (no nested loops); the rate uses merge_asof to
match each trial's registration date against same-sponsor completion dates strictly before it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INTERIM = Path(__file__).parents[3] / "data" / "interim"
_SPONSOR_PATH = _INTERIM / "sponsor_history.parquet"

_PHASE_MAP = {
    "1": 1,
    "phase 1": 1,
    "phase1": 1,
    "phase i": 1,
    "2": 2,
    "phase 2": 2,
    "phase2": 2,
    "phase ii": 2,
    "3": 3,
    "phase 3": 3,
    "phase3": 3,
    "phase iii": 3,
}

ESTABLISHED_THRESHOLD = 5
LARGE_THRESHOLD = 20

_OUT_COLS = [
    "nct_id",
    "sponsor_prior_trial_count",
    "sponsor_prior_phase_count",
    "sponsor_prior_completion_rate",
    "sponsor_prior_same_phase_completion_rate",
    "sponsor_is_established",
    "sponsor_is_large",
]


def _prior_over_earlier_dates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """COUNT of same-group trials registered on STRICTLY earlier dates (registration order).

    Returns a frame keyed by group_cols + ['date'] with column prior_total.
    """
    agg = (
        df.groupby(group_cols + ["date"], as_index=False, dropna=False)
        .agg(n_at_date=("_completed", "size"))
        .sort_values(group_cols + ["date"])
    )
    grp = agg.groupby(group_cols, dropna=False)
    # cumsum is inclusive of the current (distinct) date; subtract self → strictly earlier
    agg["prior_total"] = grp["n_at_date"].cumsum() - agg["n_at_date"]
    return agg[group_cols + ["date", "prior_total"]]


def _rate_outcome_known(
    df: pd.DataFrame, eligible_rate: pd.DataFrame, group_cols: list[str]
) -> tuple[pd.Series, pd.Series]:
    """(denominator, numerator) for the completion rate over same-group priors whose
    OUTCOME was known before this trial's registration: completion_date < registration.

    Returned Series are indexed by df['_row']; rows with no known-outcome prior are absent
    (→ NaN when mapped back → rate NaN → phase-median fill).
    """
    if len(eligible_rate) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty
    ev = (
        eligible_rate.groupby(group_cols + ["comp_date"], as_index=False)
        .agg(n=("_completed", "size"), c=("_completed", "sum"))
        .sort_values(group_cols + ["comp_date"])
    )
    g = ev.groupby(group_cols)
    ev["cum_n"] = g["n"].cumsum()  # inclusive up to this completion date
    ev["cum_c"] = g["c"].cumsum()
    ev = ev.sort_values("comp_date")
    left = df.loc[
        df["_source"].notna() & df["date"].notna(), ["_row", *group_cols, "date"]
    ].sort_values("date")
    # backward + strict → last completion strictly before the registration date
    m = pd.merge_asof(
        left,
        ev[[*group_cols, "comp_date", "cum_n", "cum_c"]],
        left_on="date",
        right_on="comp_date",
        by=group_cols,
        direction="backward",
        allow_exact_matches=False,
    )
    denom = pd.Series(m["cum_n"].to_numpy(), index=m["_row"].to_numpy())
    num = pd.Series(m["cum_c"].to_numpy(), index=m["_row"].to_numpy())
    return denom, num


def compute_sponsor_history(studies_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trial sponsor track record from prior same-sponsor trials only.

    Input columns: nct_id, source, study_first_posted_date, overall_status, phase, and
    (for leakage-safe rates) completion_date. A missing completion_date column is treated
    as all-null (rates fall back to the phase median; counts are unaffected).
    """
    df = studies_df.copy().reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["study_first_posted_date"], errors="coerce")
    if "completion_date" in df.columns:
        df["comp_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    else:
        df["comp_date"] = pd.NaT
    df["_completed"] = (df["overall_status"].astype(str).str.upper() == "COMPLETED").astype(int)
    df["_phase"] = (
        (df["phase"].fillna("").astype(str).str.lower().str.strip().map(_PHASE_MAP))
        .fillna(-1)
        .astype(int)
    )
    src = df["source"].astype("string")
    df["_source"] = src.where(src.str.strip().fillna("") != "", other=pd.NA)

    # ── COUNT features (registration order — unchanged, valid) ────────────────
    eligible = df[df["_source"].notna() & df["date"].notna()].copy()
    allp = _prior_over_earlier_dates(eligible, ["_source"])
    df = df.merge(allp, on=["_source", "date"], how="left")
    samep = _prior_over_earlier_dates(eligible, ["_source", "_phase"]).rename(
        columns={"prior_total": "phase_total"}
    )
    df = df.merge(samep, on=["_source", "_phase", "date"], how="left")
    no_hist = df["_source"].isna() | df["date"].isna()
    df.loc[no_hist, ["prior_total", "phase_total"]] = 0
    df["sponsor_prior_trial_count"] = df["prior_total"].fillna(0).astype(int)
    df["sponsor_prior_phase_count"] = df["phase_total"].fillna(0).astype(int)

    # ── RATE features (outcome-known order — ITEM 3 fix) ──────────────────────
    eligible_rate = df[df["_source"].notna() & df["comp_date"].notna()].copy()
    denom_all, num_all = _rate_outcome_known(df, eligible_rate, ["_source"])
    denom_ph, num_ph = _rate_outcome_known(df, eligible_rate, ["_source", "_phase"])
    rd = df["_row"].map(denom_all)
    rn = df["_row"].map(num_all)
    rdp = df["_row"].map(denom_ph)
    rnp = df["_row"].map(num_ph)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["sponsor_prior_completion_rate"] = np.where(rd > 0, rn / rd, np.nan)
        df["sponsor_prior_same_phase_completion_rate"] = np.where(rdp > 0, rnp / rdp, np.nan)

    # Fill no-known-history rates with the phase-level median (PHASE2.md rule).
    for rate_col in ["sponsor_prior_completion_rate", "sponsor_prior_same_phase_completion_rate"]:
        med = df.groupby("_phase")[rate_col].transform("median")
        df[rate_col] = df[rate_col].fillna(med).fillna(df[rate_col].median()).fillna(0.0)

    df["sponsor_is_established"] = (
        df["sponsor_prior_trial_count"] >= ESTABLISHED_THRESHOLD
    ).astype(int)
    df["sponsor_is_large"] = (df["sponsor_prior_trial_count"] >= LARGE_THRESHOLD).astype(int)

    result = df[_OUT_COLS].reset_index(drop=True)

    # Explicit temporal gates on a sample (the build pipeline re-runs both on ≥500 test rows):
    # (1) counts never include a future-registered trial; (2) rates never include a prior
    # whose completion was not known at registration.
    sample = _sample_nct_ids(result, n=300, seed=0)
    assert_no_future_leakage(result, studies_df, sample, context="compute_sponsor_history")
    assert_rate_outcome_known(result, studies_df, sample, context="compute_sponsor_history")

    logger.info(
        "compute_sponsor_history: %d trials, prior_count mean=%.2f max=%d, "
        "completion_rate non-fill fraction=%.3f",
        len(result),
        result["sponsor_prior_trial_count"].mean(),
        int(result["sponsor_prior_trial_count"].max()),
        float((rd > 0).mean()),
    )
    return result


def _sample_nct_ids(result: pd.DataFrame, n: int, seed: int) -> list[str]:
    k = min(n, len(result))
    return result.sample(k, random_state=seed)["nct_id"].tolist()


def assert_no_future_leakage(
    result: pd.DataFrame, studies_df: pd.DataFrame, check_nct_ids: list[str], context: str = ""
) -> list[dict]:
    """COUNT gate: raise if any checked trial's prior_count exceeds the true number of
    same-sponsor trials registered strictly before it. Returns per-trial evidence."""
    s = studies_df[["nct_id", "source", "study_first_posted_date"]].copy()
    s["d"] = pd.to_datetime(s["study_first_posted_date"], errors="coerce")
    src_arr, date_arr = s["source"].to_numpy(), s["d"].to_numpy()
    date_of = dict(zip(s["nct_id"], s["d"], strict=False))
    src_of = dict(zip(s["nct_id"], s["source"], strict=False))
    computed_of = dict(zip(result["nct_id"], result["sponsor_prior_trial_count"], strict=False))

    evidence, violations = [], []
    for nct in check_nct_ids:
        d, sp = date_of.get(nct), src_of.get(nct)
        if pd.isna(d) or pd.isna(sp):
            continue
        actual_prior = int(((src_arr == sp) & (date_arr < np.datetime64(d))).sum())
        computed = int(computed_of.get(nct, 0))
        evidence.append({"nct_id": nct, "computed_prior": computed, "actual_prior": actual_prior})
        if computed > actual_prior:
            violations.append((nct, computed, actual_prior))
    if violations:
        raise ValueError(
            f"TEMPORAL LEAKAGE in sponsor_history{' (' + context + ')' if context else ''}: "
            f"{len(violations)} trial(s) count future-registered trials as prior. "
            f"Examples: {violations[:5]}"
        )
    return evidence


def assert_rate_outcome_known(
    result: pd.DataFrame,
    studies_df: pd.DataFrame,
    check_nct_ids: list[str],
    context: str = "",
    tol: float = 1e-6,
) -> list[dict]:
    """RATE gate (ITEM 3): raise if the stored sponsor_prior_completion_rate differs from
    the honest rate recomputed over ONLY priors whose completion_date < registration date.
    A leak (counting a prior with completion on/after registration) would change the rate
    and trip this. Returns per-trial (stored vs honest) evidence for trials with ≥1 known
    prior (median-filled trials, with 0 known priors, are skipped)."""
    if "completion_date" not in studies_df.columns:
        return []  # no completion dates → all rates are phase-median fill; nothing to verify
    s = studies_df[
        ["nct_id", "source", "study_first_posted_date", "completion_date", "overall_status"]
    ].copy()
    s["R"] = pd.to_datetime(s["study_first_posted_date"], errors="coerce")
    s["C"] = pd.to_datetime(s["completion_date"], errors="coerce")
    s["done"] = (s["overall_status"].astype(str).str.upper() == "COMPLETED").astype(int)
    src_arr, cd_arr, done_arr = s["source"].to_numpy(), s["C"].to_numpy(), s["done"].to_numpy()
    reg_of = dict(zip(s["nct_id"], s["R"], strict=False))
    src_of = dict(zip(s["nct_id"], s["source"], strict=False))
    rate_of = dict(zip(result["nct_id"], result["sponsor_prior_completion_rate"], strict=False))

    evidence, violations = [], []
    for nct in check_nct_ids:
        R, sp = reg_of.get(nct), src_of.get(nct)
        if pd.isna(R) or pd.isna(sp):
            continue
        known = (src_arr == sp) & (cd_arr < np.datetime64(R))  # outcome known before registration
        denom = int(known.sum())
        if denom == 0:
            continue  # rate is phase-median-filled; nothing to compare
        honest = float(done_arr[known].sum()) / denom
        stored = float(rate_of.get(nct, np.nan))
        evidence.append(
            {"nct_id": nct, "stored_rate": stored, "honest_rate": honest, "n_known": denom}
        )
        if not np.isfinite(stored) or abs(stored - honest) > tol:
            violations.append((nct, round(stored, 4), round(honest, 4), denom))
    if violations:
        raise ValueError(
            f"RATE OUTCOME-LEAK in sponsor_history{' (' + context + ')' if context else ''}: "
            f"stored completion_rate != honest (outcome-known-before) rate for "
            f"{len(violations)} trial(s). Examples (nct, stored, honest, n_known): {violations[:5]}"
        )
    return evidence


def load_sponsor_history() -> pd.DataFrame:
    if not _SPONSOR_PATH.exists():
        raise FileNotFoundError(
            f"{_SPONSOR_PATH} not found. Run `python -m cto.pipelines.build_sponsor_history` "
            "(or `dvc repro sponsor_history`) first."
        )
    return pd.read_parquet(_SPONSOR_PATH)
