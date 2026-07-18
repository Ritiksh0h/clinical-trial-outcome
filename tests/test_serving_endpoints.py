"""Serving endpoint tests. /health, input validation, and the HTML page need no model and
run in CI; the live /predict test needs the DVC-tracked model file and skips when absent."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cto.serving.app import app
from cto.serving.predict import _TIER_THRESHOLDS, completion_tier

_MODEL3 = Path(__file__).parents[1] / "models" / "gold_phase3.joblib"
client = TestClient(app)

_TIERS = {
    "Lower completion likelihood",
    "Moderate completion likelihood",
    "Higher completion likelihood",
}


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["models_loaded"], list)


def test_index_page_states_completion_not_efficacy():
    r = client.get("/")
    assert r.status_code == 200
    html = r.text.lower()
    assert "complete" in html and "not" in html and "efficacy" in html  # honesty banner present


def test_predict_rejects_bad_phase():
    r = client.post("/predict", json={"phase": 9, "eligibility_criteria": "x"})
    assert r.status_code == 422


def test_predict_rejects_missing_phase():
    r = client.post("/predict", json={"eligibility_criteria": "x"})
    assert r.status_code == 422


def test_tier_thresholds_rank_by_phase():
    """Above the phase's 67th pctile → Higher; below the 33rd → Lower; between → Moderate,
    using the bundled per-phase thresholds. No model needed (runs in CI)."""
    for phase in (1, 2, 3):
        t = _TIER_THRESHOLDS[str(phase)]
        assert completion_tier(t["p67"] + 0.01, phase) == "Higher completion likelihood"
        assert completion_tier(t["p33"] - 0.001, phase) == "Lower completion likelihood"
        assert completion_tier((t["p33"] + t["p67"]) / 2, phase) == "Moderate completion likelihood"


@pytest.mark.skipif(not _MODEL3.exists(), reason="requires DVC-tracked model file (absent in CI)")
def test_predict_returns_tier_and_secondary_probability():
    payload = {
        "phase": 3,
        "eligibility_criteria": "Inclusion Criteria: adults aged 18 years or older. Exclusion Criteria: pregnancy.",
        "sponsor_class": "INDUSTRY",
        "allocation": "RANDOMIZED",
        "masking": "DOUBLE",
        "primary_purpose": "TREATMENT",
        "intervention_types": ["DRUG"],
        "enrollment": 250,
        "number_of_arms": 2,
        "registration_year": 2021,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    # PRIMARY — phase-relative tier
    assert body["completion_tier"] in _TIERS
    assert body["phase"] == "III"
    assert body["phase_model"] == "gold_phase3"
    assert body["tier_context"]
    # SECONDARY — raw calibrated probability + its caveat
    assert 0.0 <= body["raw_calibrated_probability"] <= 1.0
    assert "ranking" in body["probability_caveat"].lower()
    assert body["model_version"]
    # the returned tier must equal re-tiering the returned raw probability
    assert body["completion_tier"] == completion_tier(body["raw_calibrated_probability"], 3)
