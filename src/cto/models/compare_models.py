"""
Three-way model comparison (Track A, gold-only, 438 features, schema 2.1.0).

XGBoost (champion, retrained identically) vs LightGBM, CatBoost(numeric), CatBoost(native
categoricals + has_time) — a fair bake-off on the FROZEN gold test set. Same matrices, same
frozen test set (never recompute the split), same gold class weights (scale_pos_weight),
same train/val/test usage, Platt calibration on val for all.

Promotion gate (gate.py): a challenger is promoted over XGBoost only if it beats it on BOTH
PR-AUC and AUROC with the interval test passing (Phase I: walk-forward + Nadeau-Bengio;
Phase II/III: Boyd logit-CI single split) — never on a bare point difference.

CatBoost native mode uses ordered target statistics on the design/sponsor/ta_bucket
categoricals with has_time=True — valid because the gold train matrix is completion-date
sorted (featurize_gold), so TS use only earlier-completing trials (outcome-known).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cto.models.train_gold import _XGB, _load, _params

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[3]
_MODELS = _ROOT / "models"
_RAW = _ROOT / "data" / "raw"

# ordinal-encoded categoricals (present in the 438-feature matrix) for CatBoost native mode
CAT_COLS = [
    "phase_clean",
    "sponsor_class",
    "allocation",
    "intervention_model",
    "masking_ordinal",
    "primary_purpose",
    "intervention_type_primary",
    "gender",
    "healthy_volunteers",
    "ta_bucket",
]

CHALLENGERS = ["lightgbm", "catboost", "catboost_native"]


def _lgbm(spw, n_estimators=2000):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        n_estimators=n_estimators,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        min_child_samples=20,
        scale_pos_weight=spw,
        random_state=42,
        verbosity=-1,
    )


def _catboost(spw, iterations=2000, native=False):
    from catboost import CatBoostClassifier

    extra = {"has_time": True} if native else {}
    return CatBoostClassifier(
        depth=4,
        learning_rate=0.05,
        iterations=iterations,
        l2_leaf_reg=1.0,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        scale_pos_weight=spw,
        random_state=42,
        verbose=False,
        eval_metric="PRAUC",
        **extra,
    )


def _xgb(spw, n_estimators=None):
    from xgboost import XGBClassifier

    cfg = dict(_XGB)
    if n_estimators is not None:  # walk-forward: fixed trees, no early stop
        cfg["n_estimators"] = n_estimators
        cfg.pop("early_stopping_rounds", None)
    return XGBClassifier(scale_pos_weight=spw, **cfg)


def _as_cat(X: pd.DataFrame) -> pd.DataFrame:
    """Cast the categorical columns to string so CatBoost treats them as categories."""
    X = X.copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("int64").astype(str)
    return X


def _fit(kind, spw, Xtr, ytr, Xval=None, yval=None):
    """Train one model with early stopping on val (single-split) — returns fitted model."""
    if kind == "xgboost":
        m = _xgb(spw)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    elif kind == "lightgbm":
        import lightgbm as lgb

        m = _lgbm(spw)
        m.fit(
            Xtr,
            ytr,
            eval_set=[(Xval, yval)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
    elif kind == "catboost":
        m = _catboost(spw)
        m.fit(Xtr, ytr, eval_set=(Xval, yval), early_stopping_rounds=50, verbose=False)
    elif kind == "catboost_native":
        m = _catboost(spw, native=True)
        m.fit(
            _as_cat(Xtr),
            ytr,
            eval_set=(_as_cat(Xval), yval),
            cat_features=CAT_COLS,
            early_stopping_rounds=50,
            verbose=False,
        )
    return m


def _proba(kind, model, X):
    Xin = _as_cat(X) if kind == "catboost_native" else X
    return model.predict_proba(Xin)[:, 1]


def _evaluate_kind(kind, spw, phase):
    """Train + Platt-calibrate + evaluate one model on the frozen gold test split."""
    from cto.models.gate import auprc_logit_ci, evaluate
    from cto.models.train import compute_metrics

    Xtr, ytr = _load(phase, "train")
    Xval, yval = _load(phase, "val")
    Xte, yte = _load(phase, "test")
    model = _fit(kind, spw, Xtr, ytr, Xval, yval)

    # Platt calibration on val (reuse wrapper; feed the kind-correct probability getter)
    raw_val = _proba(kind, model, Xval)
    from sklearn.linear_model import LogisticRegression

    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(raw_val.reshape(-1, 1), yval)
    raw_te = _proba(kind, model, Xte)
    cal_te = platt.predict_proba(raw_te.reshape(-1, 1))[:, 1]

    ev = evaluate(yte, raw_te)
    n_pos = int(yte.sum())
    lo, hi = auprc_logit_ci(ev["prauc"], n_pos)
    return {
        "kind": kind,
        "prauc": ev["prauc"],
        "prauc_ci": (lo, hi),
        "auroc": ev["auroc"],
        "brier_raw": compute_metrics(yte, raw_te)["brier"],
        "brier_cal": compute_metrics(yte, cal_te)["brier"],
        "n_pos": n_pos,
        "no_skill": float(yte.mean()),
        "raw_test": raw_te,
        "y_test": yte,
        "model": model,
        "platt": platt,
    }


def _phase1_walk_forward(spw):
    """Walk-forward Phase I for all model kinds on identical folds. Returns per-kind per-fold
    probabilities + pooled PR-AUC, plus the shared y_by_fold and n_train/n_test."""
    from sklearn.metrics import average_precision_score

    from cto.features.build import _STUDIES_COLS, build_features
    from cto.features.leakage import drop_leakage_columns
    from cto.models.gate import walk_forward_folds

    gold = pd.read_parquet(_RAW / "cto_gold.parquet")
    member = set(pd.read_parquet(_RAW / "cto_phase1.parquet")["nct_id"])
    studies = drop_leakage_columns(
        pd.read_parquet(_RAW / "aact_studies_snapshot.parquet"), warn=False
    )
    gp = (
        gold[gold["nct_id"].isin(member)]
        .merge(
            studies[[c for c in _STUDIES_COLS if c in studies.columns]], on="nct_id", how="inner"
        )
        .reset_index(drop=True)
    )
    X = build_features(1, "test", df=gp)
    y = gp["y"].to_numpy()
    folds = walk_forward_folds(gp["completion_date"], y, test_years=(2021, 2022, 2023, 2024))

    kinds = ["xgboost", *CHALLENGERS]
    out = {k: {"per_fold": [], "y_by_fold": []} for k in kinds}
    n_train, n_test = [], []
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        n_train.append(len(tr))
        n_test.append(len(te))
        for k in kinds:
            if k == "xgboost":
                m = _xgb(spw, n_estimators=300)
                m.fit(X.iloc[tr], y[tr], verbose=False)
                prob = m.predict_proba(X.iloc[te])[:, 1]
            elif k == "lightgbm":
                m = _lgbm(spw, n_estimators=300)
                m.fit(X.iloc[tr], y[tr])
                prob = m.predict_proba(X.iloc[te])[:, 1]
            elif k == "catboost":
                m = _catboost(spw, iterations=300)
                m.fit(X.iloc[tr], y[tr], verbose=False)
                prob = m.predict_proba(X.iloc[te])[:, 1]
            else:  # catboost_native
                m = _catboost(spw, iterations=300, native=True)
                m.fit(_as_cat(X.iloc[tr]), y[tr], cat_features=CAT_COLS, verbose=False)
                prob = m.predict_proba(_as_cat(X.iloc[te]))[:, 1]
            out[k]["per_fold"].append(prob)
            out[k]["y_by_fold"].append(y[te])
    for k in kinds:
        py = np.concatenate(out[k]["y_by_fold"])
        pp = np.concatenate(out[k]["per_fold"])
        out[k]["pooled_prauc"] = float(average_precision_score(py, pp))
    return out, n_train, n_test


def run() -> None:
    import mlflow

    p = _params()
    schema = p["features"]["schema_version"]
    results = {}  # results[phase][kind] = metrics dict
    wf_cache = {}  # wf_cache[phase] = (wf_out, n_train, n_test)  (phase 1 only)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cto_gold")

    for phase in (1, 2, 3):
        spw = p["model"]["class_weight_by_phase"][phase]
        results[phase] = {}
        for kind in ["xgboost", *CHALLENGERS]:
            try:
                r = _evaluate_kind(kind, spw, phase)
            except Exception as exc:  # CatBoost-native may be finicky — degrade gracefully
                logger.warning("phase %d %s failed: %s", phase, kind, exc)
                results[phase][kind] = {"kind": kind, "failed": str(exc)}
                continue
            results[phase][kind] = r
            if kind in CHALLENGERS:  # log challengers; XGB champion already in MLflow
                with mlflow.start_run(run_name=f"gold_phase{phase}_{kind}"):
                    mlflow.set_tags(
                        {
                            "model_type": kind,
                            "training_labels": "gold",
                            "phase": str(phase),
                            "track": "A",
                            "bakeoff": "true",
                        }
                    )
                    mlflow.log_params(
                        {
                            "scale_pos_weight": spw,
                            "feature_schema_version": schema,
                            "n_features": _load(phase, "train")[0].shape[1],
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "test_prauc": r["prauc"],
                            "test_auroc": r["auroc"],
                            "test_brier_raw": r["brier_raw"],
                            "test_brier_cal": r["brier_cal"],
                        }
                    )
                    _MODELS.mkdir(parents=True, exist_ok=True)
                    mp = _MODELS / f"gold_phase{phase}_{kind}.joblib"
                    joblib.dump({"model": r["model"], "platt": r["platt"], "kind": kind}, mp)
                    mlflow.log_artifact(str(mp), "model")
        if phase == 1:
            wf_cache[phase] = _phase1_walk_forward(spw)

    _report(results, wf_cache, p)


def _report(results, wf_cache, p):
    from cto.models.gate import promotion_decision_single, promotion_decision_walkforward

    kinds = ["xgboost", *CHALLENGERS]
    print("\n" + "=" * 100)
    print("THREE-WAY BAKE-OFF (Track A, gold-only, 438 feats) — test-set PR-AUC [Boyd CI]  (AUROC)")
    print("=" * 100)
    print(f"{'phase':>5} " + "".join(f"{k:>24}" for k in kinds))
    for ph in (1, 2, 3):
        cells = []
        for k in kinds:
            r = results[ph][k]
            if "failed" in r:
                cells.append(f"{'FAILED':>24}")
            else:
                lo, hi = r["prauc_ci"]
                cells.append(f"{r['prauc']:.3f}[{lo:.2f},{hi:.2f}]({r['auroc']:.2f})".rjust(24))
        print(f"{ph:>5} " + "".join(cells))

    # Phase I walk-forward pooled (fair comparison)
    wf, ntr, nte = wf_cache[1]
    print("\nPhase I WALK-FORWARD pooled PR-AUC (fair — single-split n_pos=20):")
    for k in kinds:
        print(f"  {k:>18}: {wf[k]['pooled_prauc']:.3f}")

    # ── Promotion gate: challengers vs XGBoost champion ──
    print("\n" + "=" * 100)
    print(
        "PROMOTION GATE — challenger vs XGBoost champion (promote only if BOTH metrics + interval)"
    )
    print("=" * 100)
    for ph in (1, 2, 3):
        print(f"\nPhase {ph}:")
        for k in CHALLENGERS:
            r = results[ph][k]
            if "failed" in r:
                print(f"  {k:>18}: FAILED ({r['failed'][:60]})")
                continue
            if ph == 1:
                dec = promotion_decision_walkforward(
                    wf["xgboost"]["y_by_fold"],
                    wf["xgboost"]["per_fold"],
                    wf[k]["per_fold"],
                    ntr,
                    nte,
                )
            else:
                champ = results[ph]["xgboost"]
                dec = promotion_decision_single(champ["y_test"], champ["raw_test"], r["raw_test"])
            verdict = "PROMOTE" if dec["promote"] else "retain XGBoost"
            print(f"  {k:>18}: {verdict}  ({dec['reason']})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
