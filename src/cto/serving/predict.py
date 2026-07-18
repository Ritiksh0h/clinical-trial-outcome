"""Load the per-phase champion bundles and turn a raw trial into an Option-C prediction:
a phase-relative completion TIER (primary, correctly-readable) plus the raw calibrated
probability (secondary, caveated). Models load lazily (cached) so the app imports fine where
the DVC-tracked model files are absent — /health reports what is actually available."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import joblib

from cto.serving.featurize import featurize_trial
from cto.serving.schema import TrialInput

_MODELS_DIR = Path(__file__).parents[3] / "models"
_THRESHOLDS_PATH = Path(__file__).parent / "tier_thresholds.json"
MODEL_VERSION = "gold-xgboost-2.1.0"

_ROMAN = {1: "I", 2: "II", 3: "III"}

# Per-phase tercile thresholds of calibrated test-set scores (scripts/compute_tier_thresholds.py).
_TIER_THRESHOLDS = {
    k: v for k, v in json.loads(_THRESHOLDS_PATH.read_text()).items() if not k.startswith("_")
}

_PROBABILITY_CAVEAT = (
    "Calibrated to a failure-enriched evaluation set; not a natural completion rate. Interpret "
    "as a relative ranking, not an absolute chance. See methodology (reports/RESULTS.md)."
)


def _model_path(phase: int) -> Path:
    return _MODELS_DIR / f"gold_phase{phase}.joblib"


@functools.lru_cache(maxsize=3)
def _load_bundle(phase: int) -> dict:
    path = _model_path(phase)
    if not path.exists():
        raise FileNotFoundError(
            f"Model for phase {phase} not found at {path}. Run train_gold first."
        )
    return joblib.load(path)


def available_phases() -> list[int]:
    """Phases whose model file is present on disk (for /health)."""
    return [p for p in (1, 2, 3) if _model_path(p).exists()]


def completion_tier(prob: float, phase: int) -> str:
    """Tier a calibrated score against its OWN phase's tercile thresholds (scores are not
    comparable across phases). Below p33 → Lower; above p67 → Higher; between → Moderate.
    Worded as LIKELIHOOD (not risk) so the good direction is unambiguous."""
    t = _TIER_THRESHOLDS[str(phase)]
    if prob < t["p33"]:
        return "Lower completion likelihood"
    if prob > t["p67"]:
        return "Higher completion likelihood"
    return "Moderate completion likelihood"


def predict_completion(trial: TrialInput) -> dict:
    """Featurize the raw trial → calibrated probability → phase-relative tier (Option C)."""
    bundle = _load_bundle(trial.phase)
    x = featurize_trial(trial, trial.phase, bundle["features"])
    prob = float(bundle["calibrator"].predict_proba(x)[:, 1][0])
    return {
        "phase": _ROMAN[trial.phase],
        "completion_tier": completion_tier(prob, trial.phase),
        "tier_context": "relative to the evaluation set for this phase",
        "raw_calibrated_probability": prob,
        "probability_caveat": _PROBABILITY_CAVEAT,
        "model_version": MODEL_VERSION,
        "phase_model": f"gold_phase{trial.phase}",
    }
