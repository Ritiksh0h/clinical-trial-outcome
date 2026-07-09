"""
Feature builder for CTO prediction.
build_raw_joined: Phase 0 join (AACT studies + CTO labels).
build_features: Phase 1 full feature engineering.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# pandas 2.0 downcasting FutureWarning from fillna on mixed-type columns
warnings.filterwarnings("ignore", message="Downcasting object dtype")

from cto.features.leakage import assert_no_leakage, drop_leakage_columns  # noqa: E402

logger = logging.getLogger(__name__)

_RAW_DIR = Path(__file__).parents[3] / "data" / "raw"
_PARAMS_PATH = Path(__file__).parents[3] / "params.yaml"

# AACT phase strings vary; map all known forms to int
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

_STUDIES_COLS = [
    "nct_id",
    "overall_status",
    "enrollment",
    "enrollment_type",
    "number_of_arms",
    "number_of_groups",
    "source_class",
    "study_first_posted_date",
    "primary_completion_date",
    "completion_date",
    "last_update_posted_date",
    "phase",
]

# ── Ordinal / categorical encoding maps ───────────────────────────────────────

_SPONSOR_MAP = {
    "INDUSTRY": 0,
    "NIH": 1,
    "FED": 2,
    "OTHER_GOV": 3,
    "NETWORK": 4,
    "OTHER": 5,
    "UNKNOWN": 6,
}
_ALLOC_MAP = {"RANDOMIZED": 1, "NON_RANDOMIZED": 0}
_INTERV_MODEL_MAP = {
    "PARALLEL": 0,
    "CROSSOVER": 1,
    "SINGLE GROUP": 2,
    "SINGLE_GROUP": 2,
    "FACTORIAL": 3,
    "SEQUENTIAL": 4,
}
_MASKING_MAP = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
_PURPOSE_MAP = {
    "TREATMENT": 0,
    "PREVENTION": 1,
    "BASIC SCIENCE": 2,
    "BASIC_SCIENCE": 2,
    "DIAGNOSTIC": 3,
    "SUPPORTIVE CARE": 4,
    "SUPPORTIVE_CARE": 4,
    "SCREENING": 5,
}
_INTERV_TYPE_MAP = {
    "DRUG": 0,
    "BIOLOGICAL": 1,
    "DEVICE": 2,
    "BEHAVIORAL": 3,
    "PROCEDURE": 4,
    "RADIATION": 5,
    "DIETARY SUPPLEMENT": 6,
    "DIETARY_SUPPLEMENT": 6,
    "COMBINATION PRODUCT": 7,
}
_GENDER_MAP = {"ALL": 0, "FEMALE": 1, "MALE": 2}


# ── Helper functions ──────────────────────────────────────────────────────────


def _parse_age(s) -> float:
    """Parse AACT age string ("18 Years", "6 Months") to years as float."""
    if pd.isna(s):
        return float("nan")
    s = str(s).strip()
    if s.upper() in ("N/A", "NA", ""):
        return float("nan")
    m = re.match(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", s, re.IGNORECASE)
    if not m:
        return float("nan")
    val = float(m.group(1))
    unit = m.group(2).lower()
    if "year" in unit:
        return val
    if "month" in unit:
        return val / 12.0
    if "week" in unit:
        return val / 52.0
    return val / 365.25  # days


def _count_section(text: str, section_keyword: str) -> int:
    """Count bullet points in an inclusion/exclusion section of criteria text."""
    if not text:
        return 0
    lower = text.lower()
    start = lower.find(section_keyword)
    if start == -1:
        return 0
    # Find start of the next major section (the other keyword), or end of text
    other = "exclusion criter" if "inclusion" in section_keyword else "inclusion criter"
    other_pos = lower.find(other, start + len(section_keyword))
    end = other_pos if other_pos > start else len(text)
    section = text[start:end]
    return len(re.findall(r"(?:^|\n)\s*[\*\-•\d]", section))


def _load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def _map_healthy_volunteers(v) -> int:
    """AACT healthy_volunteers → 1 (accepts) / 0 (no) / -1 (unknown).

    The snapshot stores this as a boolean (True/False/None); older AACT text
    forms ("Yes"/"No") are handled too for robustness.
    """
    if pd.isna(v):
        return -1
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1"):
        return 1
    if s in ("false", "no", "n", "0"):
        return 0
    return -1


def _compute_enrollment_log(enrollment, enrollment_type) -> np.ndarray:
    """log1p(enrollment) for registration-time ESTIMATED rows only; NaN otherwise.

    AACT uses "ESTIMATED" for the planned (registration-time) enrollment and "ACTUAL"
    once the trial reports. ACTUAL is completion-time and soft-leaky (a low actual
    enrollment is partly a *consequence* of early termination, which correlates with the
    label), so it is never used — ESTIMATED only, never imputed (XGBoost handles the NaN).

    Registration-time trial size is only available for ESTIMATED rows (~1-4% of this
    completed-trial population). Recovering it for the full population would need an AACT
    enrollment-history pull — deferred; revisit only if post-Phase-2 SHAP shows trial size
    would matter. See config/leakage_blocklist.yaml (enrollment_conditional_note).
    """
    enroll = pd.to_numeric(enrollment, errors="coerce")
    estimated = pd.Series(enrollment_type).fillna("").astype(str).str.upper() == "ESTIMATED"
    return np.where(estimated, np.log1p(enroll), np.nan)


# ── Phase 0 function (kept for EDA / backwards compatibility) ─────────────────


def build_raw_joined(phase: int) -> pd.DataFrame:
    """
    Load CTO labels for `phase` and join to AACT studies snapshot on nct_id.
    Returns registration-time columns only; calls assert_no_leakage before returning.
    """
    labels_path = _RAW_DIR / f"cto_phase{phase}.parquet"
    studies_path = _RAW_DIR / "aact_studies_snapshot.parquet"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}. Run `dvc repro ingest` first.")
    if not studies_path.exists():
        raise FileNotFoundError(f"Missing {studies_path}. Run `dvc repro ingest` first.")

    labels = pd.read_parquet(labels_path)  # cols: nct_id, y
    studies_raw = pd.read_parquet(studies_path)

    # Drop any post-hoc columns that might have slipped into the AACT snapshot
    studies = drop_leakage_columns(studies_raw, warn=True)

    # Keep only needed columns (some may not exist after drop; use intersection)
    keep = [c for c in _STUDIES_COLS if c in studies.columns]
    studies = studies[keep].copy()

    # Normalise AACT phase strings and filter to target phase
    if "phase" in studies.columns:
        studies["phase_clean"] = studies["phase"].fillna("").str.lower().str.strip().map(_PHASE_MAP)
        studies = studies[studies["phase_clean"] == phase].copy()
        studies = studies.drop(columns=["phase"])

    # Inner join — only trials present in both CTO and AACT
    before_labels = len(labels)
    before_studies = len(studies)
    df = labels.merge(studies, on="nct_id", how="inner")
    logger.info(
        "build_raw_joined(phase=%d): labels=%d, studies=%d → joined=%d (%.1f%%)",
        phase,
        before_labels,
        before_studies,
        len(df),
        100 * len(df) / max(before_labels, 1),
    )

    null_rates = df.isnull().mean().round(3)
    for col, rate in null_rates[null_rates > 0].items():
        logger.info("  null rate — %s: %.1f%%", col, 100 * rate)

    assert_no_leakage(df, context=f"build_raw_joined(phase={phase})")
    return df


# ── Phase 1 full feature engineering ─────────────────────────────────────────


def build_features(
    phase: int,
    split: str,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering for a phase split.

    Args:
        phase: 1, 2, or 3
        split: "train", "val", or "test"
        df: pre-split DataFrame from build_raw_joined (pass from featurize.py
            to avoid redundant computation). If None, derives the split internally.

    Returns X (features only — no y, no nct_id, no date columns).
    """
    from cto.data.mirror import load_mirror
    from cto.features.split import make_temporal_splits
    from cto.features.text import _TFIDF_PATH, fit_tfidf, load_tfidf, transform_tfidf

    params = _load_params()
    feat_p = params["features"]
    tfidf_max = feat_p["tfidf_max_features"]
    tfidf_ngram = tuple(feat_p["tfidf_ngram_range"])

    if df is None:
        raw = build_raw_joined(phase)
        splits = make_temporal_splits(raw)
        df = splits[split]

    df = df.copy().reset_index(drop=True)

    # ── 1. Join AACT mirror tables ────────────────────────────────────────────

    # Designs (1:1)
    designs = load_mirror("designs")[
        ["nct_id", "allocation", "intervention_model", "masking", "primary_purpose"]
    ]
    df = df.merge(designs, on="nct_id", how="left")

    # Eligibilities (1:1)
    elig = load_mirror("eligibilities")[
        ["nct_id", "gender", "minimum_age", "maximum_age", "healthy_volunteers", "criteria"]
    ]
    df = df.merge(elig, on="nct_id", how="left")

    # Sponsors — lead only for sponsor_class; collaborator flags
    sponsors = load_mirror("sponsors")
    lead = (
        sponsors[sponsors["lead_or_collaborator"].str.lower() == "lead"][["nct_id", "agency_class"]]
        .drop_duplicates("nct_id")
        .rename(columns={"agency_class": "_sponsor_class_raw"})
    )
    df = df.merge(lead, on="nct_id", how="left")

    collabs = sponsors[sponsors["lead_or_collaborator"].str.lower() == "collaborator"]
    ind_ncts = set(collabs[collabs["agency_class"].str.upper() == "INDUSTRY"]["nct_id"])
    nih_ncts = set(collabs[collabs["agency_class"].str.upper() == "NIH"]["nct_id"])
    df["has_industry_collaborator"] = df["nct_id"].isin(ind_ncts).astype(int)
    df["has_nih_collaborator"] = df["nct_id"].isin(nih_ncts).astype(int)

    # Interventions — count, primary type, drug/bio flags
    interventions = load_mirror("interventions")
    itype = interventions["intervention_type"].str.upper()
    drug_ncts = set(interventions[itype == "DRUG"]["nct_id"])
    bio_ncts = set(interventions[itype == "BIOLOGICAL"]["nct_id"])

    interv_count = interventions.groupby("nct_id").size().reset_index(name="_interv_count")
    interv_primary = (
        interventions.groupby("nct_id")["intervention_type"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        .reset_index(name="_interv_type_raw")
    )
    df = df.merge(interv_count, on="nct_id", how="left")
    df = df.merge(interv_primary, on="nct_id", how="left")
    df["has_drug_intervention"] = df["nct_id"].isin(drug_ncts).astype(int)
    df["has_biological_intervention"] = df["nct_id"].isin(bio_ncts).astype(int)

    # Calculated values (1:1) — number_of_facilities only
    calc = load_mirror("calculated_values")[["nct_id", "number_of_facilities"]]
    df = df.merge(calc, on="nct_id", how="left")

    # Countries — distinct declared countries per trial. Ignore the `removed` flag
    # (a post-registration edit); count all declared countries = registration-time geography.
    countries = load_mirror("countries")
    country_counts = (
        countries.groupby("nct_id")["name"].nunique().reset_index(name="_num_countries")
    )
    df = df.merge(country_counts, on="nct_id", how="left")

    # ── 2. Numeric features ───────────────────────────────────────────────────

    # enrollment_log: registration-time ESTIMATED only — ACTUAL is completion-time soft
    # leakage (never impute). AACT's planned-enrollment value is "ESTIMATED", not "ANTICIPATED".
    df["enrollment_log"] = _compute_enrollment_log(
        df.get("enrollment", pd.Series(dtype=float)),
        df.get("enrollment_type", pd.Series("", index=df.index)),
    )

    df["number_of_arms"] = pd.to_numeric(
        df.get("number_of_arms", pd.Series(dtype=float)), errors="coerce"
    ).fillna(2)

    df["number_of_facilities"] = pd.to_numeric(df["number_of_facilities"], errors="coerce")
    # ponytail: no imputation — XGBoost handles NaN natively

    # num_countries: distinct declared countries; trials with no countries row → 0
    df["num_countries"] = (
        pd.to_numeric(df.get("_num_countries", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    criteria_text = df.get("criteria", pd.Series("", index=df.index)).fillna("")
    df["criteria_length"] = criteria_text.str.len()
    df["num_inclusion_criteria"] = criteria_text.apply(
        lambda t: _count_section(t, "inclusion criter")
    )
    df["num_exclusion_criteria"] = criteria_text.apply(
        lambda t: _count_section(t, "exclusion criter")
    )

    df["min_age_years"] = df.get("minimum_age", pd.Series(dtype=str)).apply(_parse_age)
    df["max_age_years"] = df.get("maximum_age", pd.Series(dtype=str)).apply(_parse_age)

    df["registration_year"] = pd.to_datetime(
        df.get("study_first_posted_date"), errors="coerce"
    ).dt.year.astype(float)

    # ── 3. Categorical features (ordinal-encoded) ─────────────────────────────

    df["phase_clean"] = phase  # constant within a per-phase model; kept for schema

    df["sponsor_class"] = (
        df["_sponsor_class_raw"]
        .fillna("UNKNOWN")
        .str.upper()
        .map(_SPONSOR_MAP)
        .fillna(6)
        .astype(int)
    )
    df["allocation"] = (
        df.get("allocation", pd.Series(dtype=str))
        .fillna("")
        .str.upper()
        .map(_ALLOC_MAP)
        .fillna(0)
        .astype(int)
    )
    df["intervention_model"] = (
        df.get("intervention_model", pd.Series(dtype=str))
        .fillna("")
        .str.upper()
        .map(_INTERV_MODEL_MAP)
        .fillna(5)
        .astype(int)
    )
    df["masking_ordinal"] = (
        df.get("masking", pd.Series(dtype=str))
        .fillna("NONE")
        .str.upper()
        .map(_MASKING_MAP)
        .fillna(0)
        .astype(int)
    )
    df["primary_purpose"] = (
        df.get("primary_purpose", pd.Series(dtype=str))
        .fillna("")
        .str.upper()
        .map(_PURPOSE_MAP)
        .fillna(6)
        .astype(int)
    )
    df["intervention_type_primary"] = (
        df.get("_interv_type_raw", pd.Series(dtype=str))
        .fillna("")
        .str.upper()
        .map(_INTERV_TYPE_MAP)
        .fillna(8)
        .astype(int)
    )
    df["gender"] = (
        df.get("gender", pd.Series(dtype=str))
        .fillna("ALL")
        .str.upper()
        .map(_GENDER_MAP)
        .fillna(0)
        .astype(int)
    )
    # healthy_volunteers: 1=accepts / 0=no / -1=unknown (from eligibilities join)
    df["healthy_volunteers"] = (
        df.get("healthy_volunteers", pd.Series([None] * len(df), index=df.index))
        .map(_map_healthy_volunteers)
        .astype(int)
    )

    # ── 4. Flag features ──────────────────────────────────────────────────────

    df["has_industry_lead"] = (df["sponsor_class"] == 0).astype(int)
    df["has_nih_lead"] = (df["sponsor_class"] == 1).astype(int)
    df["is_randomized"] = df["allocation"]  # already 0/1
    df["is_blinded"] = (df["masking_ordinal"] > 0).astype(int)
    # accepts_healthy_volunteers derived from the mapped categorical (fixes the prior
    # str=="yes" check, which was always 0 because the snapshot stores booleans).
    df["accepts_healthy_volunteers"] = (df["healthy_volunteers"] == 1).astype(int)
    df["is_multinational"] = (df["num_countries"] > 1).astype(int)
    df["has_combination_therapy"] = (
        df.get("_interv_count", pd.Series(0, index=df.index)).fillna(0) > 1
    ).astype(int)
    # ponytail: has_survival_endpoint and num_primary_outcomes deferred — need the
    # design_outcomes table (not in the 7-table mirror). See config/features.yaml.

    # ── 5. TF-IDF text features ───────────────────────────────────────────────

    criteria = df.get("criteria", pd.Series("", index=df.index)).fillna("")
    # Fit only on Phase 1 train (first ever call). All other splits/phases load.
    if split == "train" and not _TFIDF_PATH.exists():
        vectorizer = fit_tfidf(criteria, tfidf_max, tfidf_ngram)
    else:
        vectorizer = load_tfidf()

    tfidf_df = transform_tfidf(criteria, vectorizer)
    tfidf_df.index = df.index

    # ── 6. Assemble final feature matrix ─────────────────────────────────────

    feature_cols = [
        # numeric
        "enrollment_log",
        "number_of_arms",
        "number_of_facilities",
        "num_countries",
        "criteria_length",
        "num_inclusion_criteria",
        "num_exclusion_criteria",
        "min_age_years",
        "max_age_years",
        "registration_year",
        # categorical
        "phase_clean",
        "sponsor_class",
        "allocation",
        "intervention_model",
        "masking_ordinal",
        "primary_purpose",
        "intervention_type_primary",
        "gender",
        "healthy_volunteers",
        # flags
        "has_industry_lead",
        "has_nih_lead",
        "has_industry_collaborator",
        "has_nih_collaborator",
        "is_randomized",
        "is_blinded",
        "is_multinational",
        "accepts_healthy_volunteers",
        "has_drug_intervention",
        "has_biological_intervention",
        "has_combination_therapy",
    ]

    X = pd.concat(
        [df[feature_cols].reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1
    )

    assert_no_leakage(X, context=f"build_features(phase={phase}, split={split})")
    logger.info(
        "build_features(phase=%d, split=%s): %d rows, %d features",
        phase,
        split,
        len(X),
        X.shape[1],
    )
    return X
