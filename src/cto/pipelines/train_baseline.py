"""
Train baseline XGBoost per phase, calibrate, evaluate on test + gold sets,
generate SHAP and calibration plots, register in MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROCESSED = Path(__file__).parents[3] / "data" / "processed"
_RAW = Path(__file__).parents[3] / "data" / "raw"
_FIGURES = Path(__file__).parents[3] / "reports" / "figures"

# AACT phase normalisation (same map as build.py)
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


def _build_gold_features(phase: int) -> tuple[pd.DataFrame, pd.Series] | None:
    """Return (X_gold, y_gold) for the given phase, or None if too few rows."""
    from cto.features.build import _STUDIES_COLS, build_features, drop_leakage_columns

    gold_labels = pd.read_parquet(_RAW / "cto_gold.parquet")  # nct_id, y
    studies_raw = pd.read_parquet(_RAW / "aact_studies_snapshot.parquet")
    studies = drop_leakage_columns(studies_raw, warn=False)

    keep = [c for c in _STUDIES_COLS if c in studies.columns]
    studies = studies[keep].copy()

    if "phase" in studies.columns:
        studies["phase_clean"] = studies["phase"].fillna("").str.lower().str.strip().map(_PHASE_MAP)
        studies = studies[studies["phase_clean"] == phase].drop(columns=["phase"])

    gold = gold_labels.merge(studies, on="nct_id", how="inner")
    if len(gold) < 50:
        logger.warning("Gold set for phase %d has only %d rows — skipping", phase, len(gold))
        return None

    # Use split="test" so TF-IDF is loaded (never fit on gold)
    X_gold = build_features(phase, "test", df=gold)
    y_gold = gold["y"].reset_index(drop=True)
    logger.info("Gold set phase %d: %d rows (pos=%.2f)", phase, len(gold), y_gold.mean())
    return X_gold, y_gold


def run() -> None:
    import mlflow
    import mlflow.sklearn

    from cto.models.calibrate import calibrate_model, plot_calibration
    from cto.models.explain import compute_shap, get_top_features, plot_shap_summary
    from cto.models.train import compute_metrics, train_phase

    _FIGURES.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    results = {}

    for phase in [1, 2, 3]:
        logger.info("=== Phase %d ===", phase)

        # 1. Train
        run_obj, xgb_model = train_phase(phase)

        # 2. Load val/test splits
        X_val = pd.read_parquet(_PROCESSED / f"features_phase{phase}_val.parquet")
        y_val = pd.read_parquet(_PROCESSED / f"labels_phase{phase}_val.parquet")["y"].values
        X_test = pd.read_parquet(_PROCESSED / f"features_phase{phase}_test.parquet")
        y_test = pd.read_parquet(_PROCESSED / f"labels_phase{phase}_test.parquet")["y"].values

        # 3. Calibrate on val set
        calibrated = calibrate_model(xgb_model, X_val, y_val)

        # 4. Evaluate on test set
        y_prob_test = calibrated.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_prob_test)

        # 5. Evaluate on gold set (evaluation-only — never used in training)
        gold_result = _build_gold_features(phase)
        gold_metrics = {}
        if gold_result is not None:
            X_gold, y_gold = gold_result
            y_prob_gold = calibrated.predict_proba(X_gold)[:, 1]
            gold_metrics = compute_metrics(y_gold.values, y_prob_gold)

        # 6. Log test + gold metrics; save calibrated model as joblib artifact
        import tempfile

        import joblib

        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = f"{tmpdir}/calibrated_model.joblib"
            joblib.dump(calibrated, cal_path)
            with mlflow.start_run(run_id=run_obj.info.run_id):
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                if gold_metrics:
                    mlflow.log_metrics({f"gold_{k}": v for k, v in gold_metrics.items()})
                # ponytail: save as artifact (not logged model) to avoid skops trust issue
                mlflow.log_artifact(cal_path, "calibrated_model")

        # 7. Register the XGBoost model (raw) with @challenger alias
        #    Calibration is applied at serving time by loading the joblib artifact.
        model_uri = f"runs:/{run_obj.info.run_id}/model"
        mv = mlflow.register_model(model_uri, f"cto_phase{phase}")
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(f"cto_phase{phase}", "challenger", mv.version)
        logger.info("Registered cto_phase%d v%s @challenger", phase, mv.version)

        # 8. Calibration plot
        y_prob_raw_val = xgb_model.predict_proba(X_val)[:, 1]
        y_prob_cal_val = calibrated.predict_proba(X_val)[:, 1]
        plot_calibration(y_val, y_prob_raw_val, y_prob_cal_val, phase, _FIGURES)

        # 9. SHAP (on raw model — calibration doesn't change feature importance)
        shap_values, X_sample = compute_shap(xgb_model, X_test)
        plot_shap_summary(shap_values, X_sample, phase, _FIGURES)

        top = get_top_features(shap_values, list(X_test.columns))
        logger.info("Top features phase %d: %s", phase, [f["feature"] for f in top[:5]])

        # Check for suspected leakage (mean |SHAP| > 0.4 for any single feature)
        for f in top:
            if f["mean_abs_shap"] > 0.4:
                logger.warning(
                    "SUSPECTED LEAKAGE: feature '%s' has mean |SHAP|=%.3f > 0.4 "
                    "for phase %d — audit immediately",
                    f["feature"],
                    f["mean_abs_shap"],
                    phase,
                )

        results[phase] = {
            "val_prauc": compute_metrics(y_val, xgb_model.predict_proba(X_val)[:, 1])["prauc"],
            "test_prauc": test_metrics["prauc"],
            "gold_prauc": gold_metrics.get("prauc", float("nan")),
        }

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Phase':<10} {'Val PR-AUC':>12} {'Test PR-AUC':>12} {'Gold PR-AUC':>12}")
    print("-" * 65)
    phase_names = {1: "Phase I", 2: "Phase II", 3: "Phase III"}
    for phase, r in results.items():
        print(
            f"{phase_names[phase]:<10} {r['val_prauc']:>12.3f} "
            f"{r['test_prauc']:>12.3f} {r['gold_prauc']:>12.3f}"
        )
    print("=" * 65)
    print("\nBaselines to beat (PyTrial XGBoost):")
    print("  Phase I:   0.513  Phase II: 0.586  Phase III: 0.697")

    # Phase III gate check
    p3_prauc = results[3]["test_prauc"]
    if p3_prauc < 0.60:
        logger.warning(
            "Phase III test PR-AUC=%.3f is below 0.60 threshold. "
            "Audit enrollment_type filter, date column leakage, and TF-IDF fit logic.",
            p3_prauc,
        )
    else:
        logger.info("Phase III gate PASS: test PR-AUC=%.3f >= 0.60", p3_prauc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
