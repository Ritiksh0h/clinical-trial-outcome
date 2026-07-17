#!/usr/bin/env python
"""
Visualize Track A (gold-only) XGBoost results → reports/figures/model_results/.

Metrics come from the MLflow `cto_gold` runs; SHAP is recomputed from the saved
models/gold_phase{N}.joblib on the gold train matrices; the Phase I walk-forward per-fold
numbers are the deterministic outputs of train_gold (seed=42). Run headlessly:
    python scripts/visualize_model_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

ROOT = Path(__file__).parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "figures" / "model_results"
PHASES = [1, 2, 3]
PNAME = {1: "Phase I", 2: "Phase II", 3: "Phase III"}

# Phase I walk-forward on the clean 438-feature model (schema 2.1.0). Per-fold PR-AUC from the
# fit diagnostic; POOLED (0.308) matches the walkforward_pooled_prauc train_gold logs to MLflow.
# Single-split n_pos=20 is unreliable; the pooled walk-forward is the trustworthy Phase I signal.
WF = {
    2021: (0.378, 664, 168),
    2022: (0.354, 778, 155),
    2023: (0.271, 743, 119),
    2024: (0.170, 388, 19),
}
WF_POOLED = (0.308, 2573, 461)

# No "Trial size (facilities)" / "Geography" families: number_of_facilities, num_countries and
# is_multinational were dropped as conduct-accrued soft leakage, so the clean 438-feature model
# has no such features. Listing them would imply the model uses leak features it does not.
FAMILY_COLORS = {
    "Sponsor history": "#4C72B0",
    "Indication (TA) history": "#DD8452",
    "TF-IDF text": "#8C8C8C",
    "Enrollment": "#CCB974",
    "Design/eligibility": "#937860",
}


def _family(f: str) -> str:
    if f.startswith("tfidf_"):
        return "TF-IDF text"
    if f.startswith("sponsor_"):
        return "Sponsor history"
    if f.startswith("ta_"):
        return "Indication (TA) history"
    if f == "enrollment_log":
        return "Enrollment"
    return "Design/eligibility"


def _mlflow_metrics() -> dict:
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(experiment_names=["cto_gold"])
    # Select ONLY the current champion: XGBoost on the clean schema-2.1.0 (438-feature) matrix.
    # The store also holds the stale schema-2.0.0 / 441-feature runs (facilities/countries not
    # yet dropped) and the challenger runs (LightGBM/CatBoost — no CI / no-skill metrics logged).
    # Picking by recency or first-match grabbed the wrong run — that is why these figures went
    # stale. Filter explicitly on model_type + schema; keep the most recent match per phase.
    champ = runs[
        (runs["tags.model_type"] == "xgboost")
        & (runs["params.feature_schema_version"].astype(str) == "2.1.0")
    ].sort_values("start_time")  # ascending → dict overwrite keeps the most recent per phase
    out = {}
    for _, r in champ.iterrows():
        ph = int(r["tags.phase"])
        out[ph] = {
            "prauc": r["metrics.test_prauc"],
            "ci_lo": r["metrics.test_prauc_ci_low"],
            "ci_hi": r["metrics.test_prauc_ci_high"],
            "auroc": r["metrics.test_auroc"],
            "brier_raw": r["metrics.test_brier_raw"],
            "brier_cal": r["metrics.test_brier_cal"],
            "no_skill": r["metrics.no_skill_baseline"],
        }
    assert set(out) == {1, 2, 3}, (
        f"expected schema-2.1.0 xgboost runs for phases 1/2/3, got {sorted(out)} — "
        "re-run train_gold or check the cto_gold MLflow experiment"
    )
    return out


def _shap_meanabs(phase: int) -> pd.Series:
    from cto.models.explain import compute_shap

    model = joblib.load(MODELS / f"gold_phase{phase}.joblib")["model"]
    X = pd.read_parquet(PROC / f"features_gold_phase{phase}_train.parquet")
    sv, sample = compute_shap(model, X)
    return pd.Series(np.abs(sv).mean(axis=0), index=sample.columns)


def chart_prauc(m: dict) -> None:
    x = np.arange(len(PHASES))
    fig, ax = plt.subplots(figsize=(11, 7))
    prauc = [m[p]["prauc"] for p in PHASES]
    lo = [m[p]["prauc"] - m[p]["ci_lo"] for p in PHASES]
    hi = [m[p]["ci_hi"] - m[p]["prauc"] for p in PHASES]
    ax.bar(x, prauc, 0.55, yerr=[lo, hi], capsize=8, color="#4C72B0", label="PR-AUC (gold test)")
    for i, p in enumerate(PHASES):
        ax.plot([i - 0.32, i + 0.32], [m[p]["no_skill"]] * 2, "k--", lw=2)
        ax.text(
            i + 0.34, m[p]["no_skill"], f"no-skill {m[p]['no_skill']:.2f}", va="center", fontsize=10
        )
        ax.text(
            i,
            prauc[i] + hi[i] + 0.03,
            f"{prauc[i]:.3f}\nAUROC {m[p]['auroc']:.2f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PNAME[p] for p in PHASES])
    ax.set_ylabel("PR-AUC (average precision)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title(
        "Track A gold-only XGBoost: PR-AUC vs no-skill baseline\n"
        "(bar above the dashed line = real signal; error bars = Boyd 95% CI)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "1_prauc_vs_noskill.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_walkforward(m: dict) -> None:
    years = [2021, 2022, 2023, 2024]
    x = np.arange(len(years) + 1)
    vals = [WF[y][0] for y in years] + [WF_POOLED[0]]
    ns = [WF[y][2] / WF[y][1] for y in years] + [WF_POOLED[2] / WF_POOLED[1]]
    colors = ["#DD8452"] * 4 + ["#4C72B0"]
    labels = [str(y) for y in years] + ["POOLED"]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(x, vals, 0.6, color=colors)
    for i in range(len(x)):
        ax.plot([x[i] - 0.3, x[i] + 0.3], [ns[i]] * 2, "k--", lw=1.5)
        pos = WF[years[i]][2] if i < 4 else WF_POOLED[2]
        ax.text(x[i], vals[i] + 0.02, f"{vals[i]:.3f}", ha="center", fontweight="bold", fontsize=12)
        ax.text(x[i], 0.01, f"pos={pos}", ha="center", fontsize=9, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 0.55)
    ax.set_title(
        "Phase I walk-forward CV (test year on x) — the trustworthy Phase I signal\n"
        "single 2024 split has only 20 positives; POOLED (461 pos) is the summary. "
        "Dashed = per-fold no-skill",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "2_phase1_walkforward.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_shap(shap: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for ax, p in zip(axes, PHASES, strict=True):
        top = shap[p].sort_values(ascending=False).head(12)[::-1]
        fams = [_family(f) for f in top.index]
        ax.barh(range(len(top)), top.values, color=[FAMILY_COLORS[f] for f in fams])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=10)
        ax.set_title(f"{PNAME[p]} — top 12 by mean |SHAP|", fontsize=13, fontweight="bold")
        ax.set_xlabel("mean |SHAP|")
    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "What drives each model — SHAP feature importance, colored by feature family",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "3_shap_top_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_family(shap: dict) -> None:
    fams = list(FAMILY_COLORS)
    x = np.arange(len(PHASES))
    fig, ax = plt.subplots(figsize=(11, 7))
    bottom = np.zeros(len(PHASES))
    for fam in fams:
        share = []
        for p in PHASES:
            s = shap[p]
            tot = s.sum()
            share.append(s[[f for f in s.index if _family(f) == fam]].sum() / tot if tot else 0)
        ax.bar(x, share, 0.55, bottom=bottom, label=fam, color=FAMILY_COLORS[fam])
        for i in range(len(PHASES)):
            if share[i] >= 0.06:
                ax.text(
                    i,
                    bottom[i] + share[i] / 2,
                    f"{share[i]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
        bottom += np.array(share)
    ax.set_xticks(x)
    ax.set_xticklabels([PNAME[p] for p in PHASES])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of total mean |SHAP|")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=10)
    ax.set_title(
        "Feature-family composition of each model\n"
        "(eligibility text leads all phases; TA history prominent in Phase I, sponsor in Phase III)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "4_shap_family_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_calibration(m: dict) -> None:
    x = np.arange(len(PHASES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 7))
    b1 = ax.bar(
        x - w / 2, [m[p]["brier_raw"] for p in PHASES], w, label="Brier raw", color="#C44E52"
    )
    b2 = ax.bar(
        x + w / 2,
        [m[p]["brier_cal"] for p in PHASES],
        w,
        label="Brier calibrated (Platt/val)",
        color="#55A868",
    )
    for bars in (b1, b2):
        for r in bars:
            ax.text(
                r.get_x() + r.get_width() / 2,
                r.get_height() + 0.003,
                f"{r.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([PNAME[p] for p in PHASES])
    ax.set_ylabel("Brier score (lower = better calibrated)")
    ax.legend(loc="upper right", frameon=True)
    ax.set_title(
        "Calibration improvement — Platt (sigmoid) on val\n(rank metrics PR-AUC/AUROC unchanged; only calibration improves)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "5_calibration_brier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = _mlflow_metrics()
    shap = {p: _shap_meanabs(p) for p in PHASES}
    chart_prauc(m)
    chart_walkforward(m)
    chart_shap(shap)
    chart_family(shap)
    chart_calibration(m)
    print(f"Saved 5 charts to {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
