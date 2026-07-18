"""Compute per-phase completion-tier thresholds for the serving layer.

Terciles (33rd / 67th percentiles) of the CHAMPION model's *calibrated* predicted
probabilities on the FROZEN gold TEST matrices (features_gold_phase{N}_test.parquet). A live
trial's calibrated score is ranked against its phase's own distribution:
  score < p33  -> "Lower completion likelihood"
  p33..p67     -> "Moderate completion likelihood"
  score > p67  -> "Higher completion likelihood"

Why terciles of the test-set scores: the raw calibrated probability is calibrated to the
failure-enriched gold set and is not a natural completion rate, but the model's real strength
is rank-ordering (PR-AUC/AUROC are ranking metrics). Ranking a new trial against the evaluation
distribution is the honest, correctly-readable output.

Writes src/cto/serving/tier_thresholds.json. Re-run after any model retrain.
    uv run python scripts/compute_tier_thresholds.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).parents[1]
_OUT = _ROOT / "src" / "cto" / "serving" / "tier_thresholds.json"


def main() -> None:
    thresholds: dict[str, dict[str, float]] = {}
    for phase in (1, 2, 3):
        bundle = joblib.load(_ROOT / f"models/gold_phase{phase}.joblib")
        x_test = pd.read_parquet(
            _ROOT / f"data/processed/features_gold_phase{phase}_test.parquet"
        ).reindex(columns=bundle["features"])
        proba = bundle["calibrator"].predict_proba(x_test)[:, 1]
        thresholds[str(phase)] = {
            "p33": float(np.percentile(proba, 33)),
            "p67": float(np.percentile(proba, 67)),
            "n_test": int(len(proba)),
        }

    payload = {
        "_comment": (
            "Per-phase tercile thresholds = 33rd/67th percentiles of the champion model's "
            "CALIBRATED predicted probabilities on the frozen gold TEST matrices "
            "(features_gold_phase{N}_test.parquet). A new trial's calibrated score is tiered "
            "against its own phase's distribution. Regenerate with "
            "scripts/compute_tier_thresholds.py after any retrain."
        ),
        **thresholds,
    }
    _OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
