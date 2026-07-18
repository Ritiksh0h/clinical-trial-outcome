"""Turn one raw trial input into the exact 438-feature vector the trained model expects.

Parity is the whole game: this MUST reproduce, feature-for-feature and name-for-name, what
cto.features.build.build_features produces for the same trial — a mismatch is a silent wrong
prediction. To guarantee that, we reuse build.py's ACTUAL structured builders (the encoding
maps + parsers) and the SAME persisted TF-IDF vectorizer. The only intentional v1 difference:
the 11 sponsor/therapeutic-area history features use frozen population-median defaults
(history_defaults.json) instead of a live lookup — asserted separately in the parity test.
Column ORDER comes from the model bundle's own `features` list (the source of truth).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

# Reuse the real training-pipeline builders — do NOT re-implement these encodings.
from cto.features.build import (
    _ALLOC_MAP,
    _GENDER_MAP,
    _INTERV_MODEL_MAP,
    _INTERV_TYPE_MAP,
    _MASKING_MAP,
    _PURPOSE_MAP,
    _SPONSOR_MAP,
    _compute_enrollment_log,
    _count_section,
    _map_healthy_volunteers,
    _parse_age,
)
from cto.features.text import load_tfidf, transform_tfidf
from cto.serving.schema import TrialInput

_DEFAULTS_PATH = Path(__file__).parent / "history_defaults.json"

HISTORY_COLS = [
    "sponsor_prior_trial_count",
    "sponsor_prior_phase_count",
    "sponsor_prior_completion_rate",
    "sponsor_prior_same_phase_completion_rate",
    "sponsor_is_established",
    "sponsor_is_large",
    "ta_prior_trial_count",
    "ta_prior_same_phase_count",
    "ta_prior_completion_rate",
    "ta_prior_same_phase_completion_rate",
    "ta_bucket",
]

_HISTORY_DEFAULTS = {
    k: v for k, v in json.loads(_DEFAULTS_PATH.read_text()).items() if not k.startswith("_")
}


def _primary_intervention(types: list[str]) -> str | None:
    """Most common intervention type, ties broken alphabetically — matches pandas
    Series.mode().iloc[0] used in build.py."""
    clean = [t for t in types if isinstance(t, str) and t.strip()]
    if not clean:
        return None
    counts = Counter(clean)
    top = max(counts.values())
    return sorted(k for k, n in counts.items() if n == top)[0]


def featurize_trial(trial: TrialInput, phase: int, feature_order: list[str]) -> pd.DataFrame:
    """Build the single-row 438-feature matrix for `trial`, column-ordered to `feature_order`
    (the model bundle's feature list). Structured + TF-IDF features are genuinely computed;
    history features default to the per-phase population medians (v1)."""
    crit = trial.eligibility_criteria or ""
    itypes_all = list(trial.intervention_types or [])
    itypes = [t for t in itypes_all if isinstance(t, str) and t.strip()]
    primary = _primary_intervention(itypes)

    # ── encoded categoricals (same maps + fallbacks as build.py) ──
    sponsor_class = int(_SPONSOR_MAP.get((trial.sponsor_class or "UNKNOWN").upper(), 6))
    allocation = int(_ALLOC_MAP.get((trial.allocation or "").upper(), 0))
    masking_ord = int(_MASKING_MAP.get((trial.masking or "NONE").upper(), 0))
    healthy = int(_map_healthy_volunteers(trial.healthy_volunteers))

    enroll_log = _compute_enrollment_log(
        pd.Series([trial.enrollment]), pd.Series([trial.enrollment_type])
    )[0]

    row: dict[str, object] = {
        # numeric
        "enrollment_log": float(enroll_log),
        "number_of_arms": float(trial.number_of_arms) if trial.number_of_arms is not None else 2.0,
        "criteria_length": len(crit),
        "num_inclusion_criteria": _count_section(crit, "inclusion criter"),
        "num_exclusion_criteria": _count_section(crit, "exclusion criter"),
        "min_age_years": _parse_age(trial.minimum_age),
        "max_age_years": _parse_age(trial.maximum_age),
        "registration_year": (
            float(trial.registration_year) if trial.registration_year is not None else float("nan")
        ),
        # categorical
        "phase_clean": phase,
        "sponsor_class": sponsor_class,
        "allocation": allocation,
        "intervention_model": int(
            _INTERV_MODEL_MAP.get((trial.intervention_model or "").upper(), 5)
        ),
        "masking_ordinal": masking_ord,
        "primary_purpose": int(_PURPOSE_MAP.get((trial.primary_purpose or "").upper(), 6)),
        "intervention_type_primary": int(_INTERV_TYPE_MAP.get((primary or "").upper(), 8)),
        "gender": int(_GENDER_MAP.get((trial.gender or "ALL").upper(), 0)),
        "healthy_volunteers": healthy,
        # flags
        "has_industry_lead": int(sponsor_class == 0),
        "has_nih_lead": int(sponsor_class == 1),
        "has_industry_collaborator": int(bool(trial.has_industry_collaborator)),
        "has_nih_collaborator": int(bool(trial.has_nih_collaborator)),
        "is_randomized": allocation,
        "is_blinded": int(masking_ord > 0),
        "accepts_healthy_volunteers": int(healthy == 1),
        "has_drug_intervention": int(any(t.upper() == "DRUG" for t in itypes)),
        "has_biological_intervention": int(any(t.upper() == "BIOLOGICAL" for t in itypes)),
        # build.py counts intervention ROWS (incl. any with null type); use the full list length
        "has_combination_therapy": int(len(itypes_all) > 1),
    }
    row.update({c: _HISTORY_DEFAULTS[str(phase)][c] for c in HISTORY_COLS})

    struct = pd.DataFrame([row])
    tfidf = transform_tfidf(pd.Series([crit]), load_tfidf())
    X = pd.concat([struct.reset_index(drop=True), tfidf.reset_index(drop=True)], axis=1)

    missing = set(feature_order) - set(X.columns)
    if missing:
        raise ValueError(
            f"featurize_trial produced no value for {len(missing)} features: {sorted(missing)[:8]}"
        )
    return X.reindex(columns=feature_order)
