"""
Indication (therapeutic-area) history features (temporal, leakage-safe).

IDENTICAL leak profile to sponsor_history.py — the same TWO temporal windows apply, keyed
on therapeutic area (TA) instead of sponsor:

  COUNT features (ta_prior_trial_count, ta_prior_same_phase_count) use REGISTRATION order —
  a same-TA prior counts iff prior.study_first_posted_date < this.study_first_posted_date.

  RATE features (ta_prior_completion_rate, ta_prior_same_phase_completion_rate) use
  OUTCOME-KNOWN order — a same-TA prior counts toward the rate ONLY if its
  completion_date < this.study_first_posted_date. A prior registered earlier but completing
  later had no known outcome at the current registration; including it would repeat the
  Item 3 leak. Null/late completion → EXCLUDED from the rate (still counted in the count).

This module is self-contained (mirrors sponsor_history) and independently gated: its own
count + rate hard gates verify 0% leak on ≥500 real test-set trials in the build path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INTERIM = Path(__file__).parents[3] / "data" / "interim"
_INDICATION_PATH = _INTERIM / "indication_history.parquet"

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

# Therapeutic-area keyword map (substring match on lowercased MeSH terms). "OTHER" is the
# default for trials whose conditions match no bucket (or have no condition rows).
TA_KEYWORDS = {
    "ONCOLOGY": [
        "cancer",
        "tumor",
        "tumour",
        "carcinoma",
        "leukemia",
        "leukaemia",
        "lymphoma",
        "melanoma",
        "sarcoma",
        "neoplasm",
        "oncolog",
        "glioma",
        "myeloma",
        "malignan",
        "metasta",
        "adenocarcinoma",
        "blastoma",
    ],
    "CNS": [
        "alzheimer",
        "parkinson",
        "dementia",
        "schizophren",
        "depress",
        "epilep",
        "seizure",
        "migraine",
        "neurolog",
        "brain",
        "multiple sclerosis",
        "anxiety",
        "bipolar",
        "psychiatr",
        "cognit",
        "neuropath",
        "autism",
        "huntington",
    ],
    "CARDIOVASCULAR": [
        "cardiac",
        "heart",
        "hypertens",
        "coronary",
        "atrial",
        "arrhythmi",
        "stroke",
        "vascular",
        "cardiovascular",
        "myocard",
        "ischemi",
        "ischaemi",
        "angina",
        "thrombos",
        "atheroscler",
        "aortic",
    ],
    "INFECTIOUS": [
        "hiv",
        "hepatitis",
        "tuberculosis",
        "covid",
        "sars-cov",
        "infection",
        "bacterial",
        "viral",
        "virus",
        "fungal",
        "antibiotic",
        "sepsis",
        "influenza",
        "malaria",
        "candida",
    ],
    "METABOLIC": [
        "diabet",
        "obesity",
        "insulin",
        "lipid",
        "metabolic",
        "thyroid",
        "adrenal",
        "cholesterol",
        "glucose",
        "hyperglycemi",
        "dyslipidem",
        "hypoglycemi",
    ],
    "RESPIRATORY": [
        "asthma",
        "copd",
        "pulmonary",
        "respiratory",
        "bronch",
        "pneumonia",
        "cystic fibrosis",
        "dyspnea",
        "emphysema",
    ],
    "IMMUNE": [
        "rheumatoid",
        "lupus",
        "autoimmune",
        "crohn",
        "psoriasis",
        "colitis",
        "immunolog",
        "arthritis",
        "inflammatory",
        "allerg",
        "eczema",
        "dermatitis",
        "ankylosing",
    ],
}
# priority order for a trial matching multiple buckets (first wins); OTHER is the fallback.
TA_PRIORITY = [
    "ONCOLOGY",
    "CNS",
    "CARDIOVASCULAR",
    "INFECTIOUS",
    "METABOLIC",
    "RESPIRATORY",
    "IMMUNE",
]
TA_CODE = {
    "OTHER": 0,
    "ONCOLOGY": 1,
    "CARDIOVASCULAR": 2,
    "CNS": 3,
    "INFECTIOUS": 4,
    "METABOLIC": 5,
    "RESPIRATORY": 6,
    "IMMUNE": 7,
}

_OUT_COLS = [
    "nct_id",
    "ta_prior_trial_count",
    "ta_prior_same_phase_count",
    "ta_prior_completion_rate",
    "ta_prior_same_phase_completion_rate",
    "ta_bucket",
]


def _term_bucket(t: str) -> str | None:
    for b in TA_PRIORITY:
        if any(kw in t for kw in TA_KEYWORDS[b]):
            return b
    return None


def assign_therapeutic_area(nct_ids: pd.Series, conditions_df: pd.DataFrame) -> pd.Series:
    """Map each nct_id to a TA bucket name (priority-first over its MeSH terms; OTHER default)."""
    col = "downcase_mesh_term" if "downcase_mesh_term" in conditions_df.columns else "mesh_term"
    cond = conditions_df[["nct_id", col]].dropna().copy()
    cond["t"] = cond[col].astype(str).str.lower()
    term2b = {t: _term_bucket(t) for t in cond["t"].unique()}
    cond["b"] = cond["t"].map(term2b)
    cond = cond.dropna(subset=["b"])
    cond["rank"] = cond["b"].map({b: i for i, b in enumerate(TA_PRIORITY)})
    best = cond.sort_values("rank").groupby("nct_id")["b"].first()
    return nct_ids.map(best.to_dict()).fillna("OTHER")


def _prior_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """COUNT over same-group trials registered on STRICTLY earlier dates (registration order)."""
    agg = (
        df.groupby(group_cols + ["date"], as_index=False, dropna=False)
        .agg(n_at_date=("_completed", "size"))
        .sort_values(group_cols + ["date"])
    )
    agg["prior_total"] = (
        agg.groupby(group_cols, dropna=False)["n_at_date"].cumsum() - agg["n_at_date"]
    )
    return agg[group_cols + ["date", "prior_total"]]


def _rate_outcome_known(
    df: pd.DataFrame, eligible_rate: pd.DataFrame, group_cols: list[str]
) -> tuple[pd.Series, pd.Series]:
    """(denominator, numerator) over same-group priors whose OUTCOME was known before this
    trial's registration: completion_date < registration. Indexed by df['_row']."""
    if len(eligible_rate) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty
    ev = (
        eligible_rate.groupby(group_cols + ["comp_date"], as_index=False)
        .agg(n=("_completed", "size"), c=("_completed", "sum"))
        .sort_values(group_cols + ["comp_date"])
    )
    g = ev.groupby(group_cols)
    ev["cum_n"] = g["n"].cumsum()
    ev["cum_c"] = g["c"].cumsum()
    ev = ev.sort_values("comp_date")
    left = df.loc[df["date"].notna(), ["_row", *group_cols, "date"]].sort_values("date")
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


def compute_indication_history(
    studies_df: pd.DataFrame, conditions_df: pd.DataFrame
) -> pd.DataFrame:
    """Per-trial therapeutic-area track record from prior same-TA trials only.

    studies_df: nct_id, study_first_posted_date, completion_date, overall_status, phase.
    conditions_df: nct_id, (downcase_)mesh_term.
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
    df["_ta"] = assign_therapeutic_area(df["nct_id"], conditions_df)  # never NA (OTHER default)

    # ── COUNT features (registration order) ───────────────────────────────────
    eligible = df[df["date"].notna()].copy()
    allp = _prior_counts(eligible, ["_ta"])
    df = df.merge(allp, on=["_ta", "date"], how="left")
    samep = _prior_counts(eligible, ["_ta", "_phase"]).rename(
        columns={"prior_total": "phase_total"}
    )
    df = df.merge(samep, on=["_ta", "_phase", "date"], how="left")
    df.loc[df["date"].isna(), ["prior_total", "phase_total"]] = 0
    df["ta_prior_trial_count"] = df["prior_total"].fillna(0).astype(int)
    df["ta_prior_same_phase_count"] = df["phase_total"].fillna(0).astype(int)

    # ── RATE features (outcome-known order) ───────────────────────────────────
    eligible_rate = df[df["comp_date"].notna()].copy()
    denom_all, num_all = _rate_outcome_known(df, eligible_rate, ["_ta"])
    denom_ph, num_ph = _rate_outcome_known(df, eligible_rate, ["_ta", "_phase"])
    rd = df["_row"].map(denom_all)
    rn = df["_row"].map(num_all)
    rdp = df["_row"].map(denom_ph)
    rnp = df["_row"].map(num_ph)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["ta_prior_completion_rate"] = np.where(rd > 0, rn / rd, np.nan)
        df["ta_prior_same_phase_completion_rate"] = np.where(rdp > 0, rnp / rdp, np.nan)

    # Fill no-history rates with the phase-level median (same convention as sponsor history).
    for rate_col in ["ta_prior_completion_rate", "ta_prior_same_phase_completion_rate"]:
        med = df.groupby("_phase")[rate_col].transform("median")
        df[rate_col] = df[rate_col].fillna(med).fillna(df[rate_col].median()).fillna(0.0)

    df["ta_bucket"] = df["_ta"].map(TA_CODE).fillna(0).astype(int)

    result = df[_OUT_COLS].reset_index(drop=True)

    sample = result.sample(min(300, len(result)), random_state=0)["nct_id"].tolist()
    assert_no_future_leakage_ta(
        result, studies_df, conditions_df, sample, context="compute_indication_history"
    )
    assert_rate_outcome_known_ta(
        result, studies_df, conditions_df, sample, context="compute_indication_history"
    )
    logger.info(
        "compute_indication_history: %d trials, OTHER=%.1f%%, rate non-fill=%.3f",
        len(result),
        100 * (df["_ta"] == "OTHER").mean(),
        float((rd > 0).mean()),
    )
    return result


def _ta_and_dates(studies_df: pd.DataFrame, conditions_df: pd.DataFrame):
    ta = assign_therapeutic_area(studies_df["nct_id"], conditions_df).to_numpy()
    R = pd.to_datetime(studies_df["study_first_posted_date"], errors="coerce").to_numpy()
    C = pd.to_datetime(studies_df.get("completion_date"), errors="coerce").to_numpy()
    done = (
        (studies_df["overall_status"].astype(str).str.upper() == "COMPLETED").to_numpy().astype(int)
    )
    nid = studies_df["nct_id"].to_numpy()
    return nid, ta, R, C, done


def assert_no_future_leakage_ta(result, studies_df, conditions_df, check_nct_ids, context=""):
    """COUNT gate: prior_count must not exceed same-TA trials registered strictly before."""
    nid, ta, R, C, done = _ta_and_dates(studies_df, conditions_df)
    ta_of = dict(zip(nid, ta, strict=False))
    reg_of = dict(zip(nid, R, strict=False))
    computed_of = dict(zip(result["nct_id"], result["ta_prior_trial_count"], strict=False))
    evidence, violations = [], []
    for n in check_nct_ids:
        r, t = reg_of.get(n), ta_of.get(n)
        if pd.isna(r) or t is None:
            continue
        actual = int(((ta == t) & (R < np.datetime64(r))).sum())
        computed = int(computed_of.get(n, 0))
        evidence.append({"nct_id": n, "computed_prior": computed, "actual_prior": actual})
        if computed > actual:
            violations.append((n, computed, actual))
    if violations:
        raise ValueError(f"TA COUNT LEAKAGE ({context}): {violations[:5]}")
    return evidence


def assert_rate_outcome_known_ta(
    result, studies_df, conditions_df, check_nct_ids, context="", tol=1e-6
):
    """RATE gate: stored ta_prior_completion_rate == honest rate over same-TA priors whose
    completion_date < registration. Returns per-trial evidence for trials with ≥1 known prior."""
    nid, ta, R, C, done = _ta_and_dates(studies_df, conditions_df)
    ta_of = dict(zip(nid, ta, strict=False))
    reg_of = dict(zip(nid, R, strict=False))
    rate_of = dict(zip(result["nct_id"], result["ta_prior_completion_rate"], strict=False))
    evidence, violations = [], []
    for n in check_nct_ids:
        r, t = reg_of.get(n), ta_of.get(n)
        if pd.isna(r) or t is None:
            continue
        known = (ta == t) & (C < np.datetime64(r))
        denom = int(known.sum())
        if denom == 0:
            continue
        honest = float(done[known].sum()) / denom
        stored = float(rate_of.get(n, np.nan))
        evidence.append(
            {"nct_id": n, "stored_rate": stored, "honest_rate": honest, "n_known": denom}
        )
        if not np.isfinite(stored) or abs(stored - honest) > tol:
            violations.append((n, round(stored, 4), round(honest, 4), denom))
    if violations:
        raise ValueError(f"TA RATE OUTCOME-LEAK ({context}): {violations[:5]}")
    return evidence


def load_indication_history() -> pd.DataFrame:
    if not _INDICATION_PATH.exists():
        raise FileNotFoundError(
            f"{_INDICATION_PATH} not found. Run `python -m cto.pipelines.build_indication_history` "
            "(or `dvc repro indication_history`) first."
        )
    return pd.read_parquet(_INDICATION_PATH)
