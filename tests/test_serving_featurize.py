"""CRITICAL: the serving featurizer must reproduce the TRAINING feature vector.

For real gold trials we build the vector two ways — the training pipeline
(cto.features.build.build_features) and the new serving path (featurize_trial) — and assert
the 27 structured + 400 TF-IDF features are identical (same names, order, values). The 11
history features are expected to differ (v1 defaults them to population medians); that gap is
asserted explicitly. Data-dependent (AACT mirror + gold matrices) → skipped in CI.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

_ROOT = Path(__file__).parents[1]
_AACT = _ROOT / "data" / "raw" / "aact_studies_snapshot.parquet"
_MODEL = _ROOT / "models" / "gold_phase3.joblib"
_HAS_DATA = _AACT.exists() and _MODEL.exists()

pytestmark = pytest.mark.skipif(
    not _HAS_DATA, reason="requires DVC-tracked AACT mirror + trained model (absent in CI)"
)

_PHASE = 3


def _clean_gold_trials(n: int):
    """Reconstruct TrialInput objects for up to `n` real gold Phase-III trials that have the
    fields populated (criteria, lead sponsor, non-null intervention types), plus the nct_id and
    single-row raw frame each one came from (for the training-side build)."""
    from cto.data.mirror import load_mirror
    from cto.features.build import build_raw_joined
    from cto.serving.schema import TrialInput

    raw = build_raw_joined(_PHASE)
    ncts = set(raw["nct_id"])
    elig = load_mirror("eligibilities")
    elig = elig[elig["nct_id"].isin(ncts)].drop_duplicates("nct_id").set_index("nct_id")
    designs = load_mirror("designs")
    designs = designs[designs["nct_id"].isin(ncts)].drop_duplicates("nct_id").set_index("nct_id")
    sponsors = load_mirror("sponsors")
    sponsors = sponsors[sponsors["nct_id"].isin(ncts)]
    interv = load_mirror("interventions")
    interv = interv[interv["nct_id"].isin(ncts)]

    lead = sponsors[sponsors["lead_or_collaborator"].str.lower() == "lead"].set_index("nct_id")
    collab = sponsors[sponsors["lead_or_collaborator"].str.lower() == "collaborator"]
    ind_collab = set(collab[collab["agency_class"].str.upper() == "INDUSTRY"]["nct_id"])
    nih_collab = set(collab[collab["agency_class"].str.upper() == "NIH"]["nct_id"])

    def _v(frame, nct, col):
        if nct not in frame.index:
            return None
        val = frame.loc[nct, col]
        return None if pd.isna(val) else val

    out = []
    for nct in raw["nct_id"]:
        if nct not in elig.index or pd.isna(elig.loc[nct, "criteria"]):
            continue
        if nct not in lead.index:
            continue
        iv = interv[interv["nct_id"] == nct]
        if iv.empty or iv["intervention_type"].isna().any():
            continue

        s = raw[raw["nct_id"] == nct].reset_index(drop=True)
        srow = s.iloc[0]
        hv = _v(elig, nct, "healthy_volunteers")
        trial = TrialInput(
            phase=_PHASE,
            eligibility_criteria=str(elig.loc[nct, "criteria"]),
            enrollment=None if pd.isna(srow["enrollment"]) else int(srow["enrollment"]),
            enrollment_type=(
                "ESTIMATED" if pd.isna(srow["enrollment_type"]) else str(srow["enrollment_type"])
            ),
            number_of_arms=None if pd.isna(srow["number_of_arms"]) else int(srow["number_of_arms"]),
            allocation=_v(designs, nct, "allocation"),
            intervention_model=_v(designs, nct, "intervention_model"),
            masking=_v(designs, nct, "masking"),
            primary_purpose=_v(designs, nct, "primary_purpose"),
            gender=_v(elig, nct, "gender"),
            minimum_age=_v(elig, nct, "minimum_age"),
            maximum_age=_v(elig, nct, "maximum_age"),
            healthy_volunteers=None if hv is None else bool(hv),
            sponsor_class=_v(lead, nct, "agency_class"),
            has_industry_collaborator=nct in ind_collab,
            has_nih_collaborator=nct in nih_collab,
            intervention_types=iv["intervention_type"].tolist(),
            registration_year=int(pd.to_datetime(srow["study_first_posted_date"]).year),
        )
        out.append((nct, s, trial))
        if len(out) >= n:
            break
    return out


def test_serving_reproduces_training_structured_and_tfidf():
    from cto.features.build import build_features
    from cto.serving.featurize import HISTORY_COLS, featurize_trial

    feature_order = joblib.load(_MODEL)["features"]
    trials = _clean_gold_trials(6)
    assert trials, "no clean gold Phase-III trials found to test parity"

    struct_and_text = [c for c in feature_order if c not in HISTORY_COLS]

    for nct, single, trial in trials:
        x_train = build_features(_PHASE, "test", df=single)
        x_serve = featurize_trial(trial, _PHASE, feature_order)

        # column set + order match the model's feature list on both sides
        assert list(x_serve.columns) == feature_order
        assert list(x_train.columns) == feature_order, f"{nct}: training columns off"

        pd.testing.assert_frame_equal(
            x_serve[struct_and_text].reset_index(drop=True),
            x_train[struct_and_text].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-9,
            rtol=1e-6,
            obj=f"structured+tfidf parity for {nct}",
        )


def test_history_features_use_documented_v1_defaults():
    """The 11 history features are the intentional v1 gap: serving fills them with the frozen
    per-phase population medians, which will differ from the trial's true training values."""
    from cto.serving.featurize import _HISTORY_DEFAULTS, HISTORY_COLS, featurize_trial

    feature_order = joblib.load(_MODEL)["features"]
    trials = _clean_gold_trials(1)
    assert trials
    _, _, trial = trials[0]

    x_serve = featurize_trial(trial, _PHASE, feature_order)
    for col in HISTORY_COLS:
        assert x_serve[col].iloc[0] == _HISTORY_DEFAULTS[str(_PHASE)][col]


def test_featurize_produces_full_438_vector_no_nans_in_structured():
    """A sparse input still yields all 438 columns; the encoded structured block has no NaNs
    except the two features that are NaN-by-design (enrollment_log for non-ESTIMATED,
    registration_year when unknown)."""
    from cto.serving.featurize import featurize_trial
    from cto.serving.schema import TrialInput

    feature_order = joblib.load(_MODEL)["features"]
    x = featurize_trial(
        TrialInput(phase=3, eligibility_criteria="Inclusion Criteria: adults."), 3, feature_order
    )
    assert x.shape == (1, 438)
    allowed_nan = {"enrollment_log", "registration_year", "min_age_years", "max_age_years"}
    structured = [c for c in feature_order if not c.startswith("tfidf_")]
    nan_cols = {c for c in structured if pd.isna(x[c].iloc[0])}
    assert nan_cols <= allowed_nan, (
        f"unexpected NaNs in structured features: {nan_cols - allowed_nan}"
    )
