"""XGBoost baseline training, per phase."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PROCESSED = Path(__file__).parents[3] / "data" / "processed"
_PARAMS_PATH = Path(__file__).parents[3] / "params.yaml"


def _load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute PR-AUC, AUROC, F1, Brier score, and ECE."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    prauc = float(average_precision_score(y_true, y_prob))
    auroc = float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = _compute_ece(y_true, y_prob)
    return {"prauc": prauc, "auroc": auroc, "f1": f1, "brier": brier, "ece": ece}


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece / max(len(y_true), 1))


def train_phase(phase: int, experiment_name: str = "cto_baseline"):
    """Train XGBoost baseline for one phase. Returns the MLflow run."""
    import mlflow
    import mlflow.xgboost
    from xgboost import XGBClassifier

    params = _load_params()
    model_p = params["model"]

    # Use literature-derived class weights — NOT computed from training labels
    scale_pos_weight = model_p["class_weight_by_phase"][phase]

    X_train = pd.read_parquet(_PROCESSED / f"features_phase{phase}_train.parquet")
    y_train = pd.read_parquet(_PROCESSED / f"labels_phase{phase}_train.parquet")["y"].values
    X_val = pd.read_parquet(_PROCESSED / f"features_phase{phase}_val.parquet")
    y_val = pd.read_parquet(_PROCESSED / f"labels_phase{phase}_val.parquet")["y"].values

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
        "random_state": model_p["random_state"],
        "tree_method": "hist",
        "verbosity": 0,
    }

    model = XGBClassifier(**xgb_params)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"phase{phase}_baseline") as run:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob_val = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_prob_val)

        mlflow.log_params(
            {
                **xgb_params,
                "phase": str(phase),
                "feature_schema_version": params["features"]["schema_version"],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_features": X_train.shape[1],
            }
        )
        mlflow.log_metrics({k: v for k, v in metrics.items()})
        mlflow.xgboost.log_model(model, "model")

        logger.info(
            "Phase %d — val PR-AUC=%.3f  AUROC=%.3f  best_iter=%d",
            phase,
            metrics["prauc"],
            metrics["auroc"],
            model.best_iteration if hasattr(model, "best_iteration") else -1,
        )

    return run, model
