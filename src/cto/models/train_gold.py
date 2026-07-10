"""
Track A (gold-only) XGBoost training, per phase — Phase 2 Step 5.

Trains three separate XGBoost models on the gold matrices from featurize_gold, predicting
COMPLETION (1) vs termination/withdrawal (0). Gold-only (Track B dropped). XGBoost only
for now (LightGBM/CatBoost deferred pending these results).

Test set is defined ONLY by data/processed/gold_test_nct_ids.json (never recompute the
gold split). scale_pos_weight from params gold rates. Calibration = Platt/sigmoid on val
(small-n safe). Phase I additionally gets the walk-forward evaluation (gate.py) because its
single-split test has only ~20 positives.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[3]
_PROCESSED = _ROOT / "data" / "processed"
_RAW = _ROOT / "data" / "raw"
_MODELS = _ROOT / "models"
_FIGS = _ROOT / "reports" / "figures" / "gold"
_PARAMS = _ROOT / "params.yaml"

# reasonable regularized config — honest baseline, not a hyperparameter competition
_XGB = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    min_child_weight=5.0,
    gamma=0.0,
    eval_metric="aucpr",
    early_stopping_rounds=50,
    tree_method="hist",
    random_state=42,
    verbosity=0,
)


def _load(phase: int, split: str):
    X = pd.read_parquet(_PROCESSED / f"features_gold_phase{phase}_{split}.parquet")
    y = pd.read_parquet(_PROCESSED / f"labels_gold_phase{phase}_{split}.parquet")["y"].values
    return X, y


def _params():
    return yaml.safe_load(open(_PARAMS))


def train_gold_phase(phase: int) -> dict:
    import mlflow
    from xgboost import XGBClassifier

    from cto.features.contamination_guard import load_gold_test_nct_ids
    from cto.models.calibrate import calibrate_model
    from cto.models.explain import compute_shap, get_top_features, plot_shap_summary
    from cto.models.gate import auprc_logit_ci, evaluate
    from cto.models.train import compute_metrics

    p = _params()
    spw = p["model"]["class_weight_by_phase"][phase]
    schema = p["features"]["schema_version"]

    Xtr, ytr = _load(phase, "train")
    Xval, yval = _load(phase, "val")
    Xte, yte = _load(phase, "test")

    # frozen test set is authoritative — confirm the test matrix matches its count
    frozen = load_gold_test_nct_ids(phase=phase)
    assert len(Xte) == len(
        frozen
    ), f"phase {phase}: test matrix {len(Xte)} rows != frozen {len(frozen)} nct_ids"

    model = XGBClassifier(scale_pos_weight=spw, **_XGB)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

    cal = calibrate_model(model, Xval, yval, method="sigmoid")  # Platt on val

    raw_te = model.predict_proba(Xte)[:, 1]
    cal_te = cal.predict_proba(Xte)[:, 1]
    n_pos = int(yte.sum())

    # PR-AUC / AUROC are rank metrics → identical for raw vs (monotonic) calibrated.
    ev = evaluate(yte, raw_te)  # {prauc, auroc}
    ci_lo, ci_hi = auprc_logit_ci(ev["prauc"], n_pos)
    brier_raw = compute_metrics(yte, raw_te)["brier"]
    brier_cal = compute_metrics(yte, cal_te)["brier"]
    no_skill = float(yte.mean())

    # SHAP on the train set (larger, stable ranking)
    sv, sample = compute_shap(model, Xtr)
    top = get_top_features(sv, list(Xtr.columns), n=15)
    _FIGS.mkdir(parents=True, exist_ok=True)
    plot_shap_summary(sv, sample, phase, _FIGS)

    # Phase I: walk-forward is the trustworthy signal (single-split n_pos≈20 too small)
    wf = _walk_forward_phase1(spw) if phase == 1 else None

    result = {
        "phase": phase,
        "n_train": len(Xtr),
        "n_val": len(Xval),
        "n_test": len(Xte),
        "n_pos_test": n_pos,
        "no_skill": no_skill,
        "prauc": ev["prauc"],
        "prauc_ci": (ci_lo, ci_hi),
        "auroc": ev["auroc"],
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
        "top_features": top,
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
    }
    if wf:
        result["walk_forward"] = wf

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cto_gold")
    with mlflow.start_run(run_name=f"gold_phase{phase}_xgb"):
        mlflow.set_tags(
            {
                "model_type": "xgboost",
                "training_labels": "gold",
                "phase": str(phase),
                "track": "A",
                "calibration": "sigmoid_on_val",
            }
        )
        mlflow.log_params(
            {
                **{k: v for k, v in _XGB.items()},
                "scale_pos_weight": spw,
                "feature_schema_version": schema,
                "n_features": Xtr.shape[1],
                "n_train": len(Xtr),
                "n_test": len(Xte),
            }
        )
        mlflow.log_metrics(
            {
                "test_prauc": ev["prauc"],
                "test_prauc_ci_low": ci_lo,
                "test_prauc_ci_high": ci_hi,
                "test_auroc": ev["auroc"],
                "test_brier_raw": brier_raw,
                "test_brier_cal": brier_cal,
                "n_pos_test": n_pos,
                "no_skill_baseline": no_skill,
            }
        )
        if wf:
            mlflow.log_metric("walkforward_pooled_prauc", wf["pooled_prauc"])
        with tempfile.TemporaryDirectory() as td:
            _MODELS.mkdir(parents=True, exist_ok=True)
            model_path = _MODELS / f"gold_phase{phase}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "calibrator": cal,
                    "features": list(Xtr.columns),
                    "phase": phase,
                    "calibration": "sigmoid",
                },
                model_path,
            )
            shap_json = Path(td) / f"shap_top_phase{phase}.json"
            shap_json.write_text(json.dumps(top, indent=2))
            mlflow.log_artifact(str(model_path), "model")
            mlflow.log_artifact(str(shap_json), "shap")

    return result


def _walk_forward_phase1(spw: float) -> dict:
    """Walk-forward temporal CV for Phase I (test 2021/2022/2023/2024) — the trustworthy
    Phase I signal, since the single 2024 split has only ~20 positives. Rebuilds gold
    Phase I features WITH completion dates (aligned) so folds can partition by year; this is
    a separate evaluation, not a recompute of the frozen single-split test set."""
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    from cto.features.build import _STUDIES_COLS, build_features
    from cto.features.leakage import drop_leakage_columns
    from cto.models.gate import walk_forward_folds

    gold = pd.read_parquet(_RAW / "cto_gold.parquet")
    member = set(pd.read_parquet(_RAW / "cto_phase1.parquet")["nct_id"])
    studies = drop_leakage_columns(
        pd.read_parquet(_RAW / "aact_studies_snapshot.parquet"), warn=False
    )
    studies_k = studies[[c for c in _STUDIES_COLS if c in studies.columns]]
    gp = (
        gold[gold["nct_id"].isin(member)]
        .merge(studies_k, on="nct_id", how="inner")
        .reset_index(drop=True)
    )

    X = build_features(1, "test", df=gp)
    y = gp["y"].to_numpy()
    dates = gp["completion_date"]

    folds = walk_forward_folds(dates, y, test_years=(2021, 2022, 2023, 2024))
    per_fold, pooled_y, pooled_p = [], [], []
    wf_cfg = dict(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        min_child_weight=5.0,
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        m = XGBClassifier(**wf_cfg)
        m.fit(X.iloc[tr], y[tr], verbose=False)
        prob = m.predict_proba(X.iloc[te])[:, 1]
        yt = y[te]
        per_fold.append(
            {
                "year": f["test_year"],
                "n_pos": int(yt.sum()),
                "n": len(yt),
                "prauc": float(average_precision_score(yt, prob)),
            }
        )
        pooled_y.append(yt)
        pooled_p.append(prob)
    pooled_y = np.concatenate(pooled_y)
    pooled_p = np.concatenate(pooled_p)
    pooled_prauc = float(average_precision_score(pooled_y, pooled_p))
    return {
        "per_fold": per_fold,
        "pooled_prauc": pooled_prauc,
        "pooled_n_pos": int(pooled_y.sum()),
        "pooled_n": int(len(pooled_y)),
    }


def run() -> None:
    results = {}
    for phase in (1, 2, 3):
        logger.info("=== training gold phase %d ===", phase)
        results[phase] = train_gold_phase(phase)

    print("\n" + "=" * 92)
    print("TRACK A (gold-only) XGBoost — per-phase TEST-set results")
    print("=" * 92)
    print(
        f"{'phase':>5} {'n_test':>6} {'pos':>4} {'no-skill':>8} {'PR-AUC':>7} "
        f"{'PR-AUC 95% CI':>16} {'AUROC':>6} {'Brier raw→cal':>15}"
    )
    for ph in (1, 2, 3):
        r = results[ph]
        lo, hi = r["prauc_ci"]
        print(
            f"{ph:>5} {r['n_test']:>6} {r['n_pos_test']:>4} {r['no_skill']:>8.3f} "
            f"{r['prauc']:>7.3f} {'[' + format(lo, '.3f') + ',' + format(hi, '.3f') + ']':>16} "
            f"{r['auroc']:>6.3f} {format(r['brier_raw'], '.3f') + '→' + format(r['brier_cal'], '.3f'):>15}"
        )

    wf = results[1].get("walk_forward")
    if wf:
        print("\nPhase I WALK-FORWARD (trustworthy — single-split n_pos=20 is too small):")
        for f in wf["per_fold"]:
            print(f"  test {f['year']}: PR-AUC={f['prauc']:.3f}  (n={f['n']}, pos={f['n_pos']})")
        print(
            f"  POOLED PR-AUC={wf['pooled_prauc']:.3f}  (pooled n={wf['pooled_n']}, pos={wf['pooled_n_pos']})"
        )

    for ph in (1, 2, 3):
        print(f"\n--- Phase {ph} SHAP top 15 (mean |SHAP|, on train) ---")
        for f in results[ph]["top_features"]:
            print(f"  {f['feature']:<42} {f['mean_abs_shap']:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
