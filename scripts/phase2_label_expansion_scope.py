#!/usr/bin/env python
"""
Stage-1 scope: quantify the Phase II label-expansion opportunity + covariate-shift risk.
DIAGNOSTIC ONLY — no training of any model we keep, no matrix changed, no labels added.

Premise (verified in-script): the completion target = overall_status (COMPLETED=1 vs
TERMINATED/WITHDRAWN=0), which agrees with the curated gold `y` at ~97%. So Phase II trials
can be labeled mechanically straight from the AACT snapshot. This stage counts the new pool,
measures the covariate shift (domain AUC, comparable to the Track-B kill), sizes the overlap
region, and returns a go/no-go verdict.

Writes reports/phase2_label_expansion_scope.md + two figures in reports/figures/fit/.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from cto.features.build import _STUDIES_COLS, build_features
from cto.features.leakage import drop_leakage_columns

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures" / "fit"
OUT = ROOT / "reports" / "phase2_label_expansion_scope.md"

ROUTE = {"PHASE2", "PHASE1/PHASE2", "PHASE2/PHASE3"}   # Phase II CTO-membership routing
TERMINAL = {"COMPLETED", "TERMINATED", "WITHDRAWN"}
NEW_SAMPLE = 8000        # domain-classifier sample per methodology note
SEED = 42


def _gold_p2_features():
    parts = [pd.read_parquet(PROC / f"features_gold_phase2_{s}.parquet")
             for s in ("train", "val", "test")]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    S = pd.read_parquet(RAW / "aact_studies_snapshot.parquet")
    G = pd.read_parquet(RAW / "cto_gold.parquet")
    P2 = pd.read_parquet(RAW / "cto_phase2.parquet")
    gold_p2 = set(G["nct_id"]) & set(P2["nct_id"])

    # ── Part 1: counts + label agreement + base rates ─────────────────────────
    gp = (G[G["nct_id"].isin(gold_p2)]
          .merge(S[["nct_id", "overall_status"]], on="nct_id", how="left"))
    gp["mech"] = (gp["overall_status"] == "COMPLETED").astype(int)
    label_agree = float((gp["y"] == gp["mech"]).mean())
    gold_pos = float(gp["y"].mean())

    intv = S[(S["study_type"] == "INTERVENTIONAL") & (S["phase"].isin(ROUTE))].copy()
    term = intv[intv["overall_status"].isin(TERMINAL)]
    new = term[~term["nct_id"].isin(gold_p2)].copy()
    new_pos = float((new["overall_status"] == "COMPLETED").mean())
    n_new = len(new)

    # ── Part 4: censoring / hygiene ───────────────────────────────────────────
    excl = intv[~intv["overall_status"].isin(TERMINAL)]["overall_status"].value_counts()
    n_excl = int(excl.sum())
    n_missing_date = int(new["study_first_posted_date"].isna().sum())

    # ── Build features for domain classifier (gold on disk; new sampled) ──────
    Xg = _gold_p2_features()
    Sc = drop_leakage_columns(S, warn=False)
    scols = [c for c in _STUDIES_COLS if c in Sc.columns]
    new_s = new.sample(min(NEW_SAMPLE, n_new), random_state=SEED)
    new_join = Sc[Sc["nct_id"].isin(new_s["nct_id"])][scols].reset_index(drop=True)
    Xn = build_features(2, "test", df=new_join).reindex(columns=Xg.columns)

    Xdom = pd.concat([Xg, Xn], ignore_index=True)
    ydom = np.r_[np.zeros(len(Xg)), np.ones(len(Xn))]  # 0=gold, 1=new

    # ── Part 2: registration-year shift + domain AUC (Track-B-comparable) ─────
    yr = lambda ids: pd.to_datetime(  # noqa: E731
        S.set_index("nct_id").reindex(list(ids))["study_first_posted_date"]).dt.year
    gold_yr = yr(gold_p2).dropna()
    new_yr = pd.to_datetime(new["study_first_posted_date"]).dt.year.dropna()

    clf = XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.7, eval_metric="logloss", random_state=SEED,
                        tree_method="hist", verbosity=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clf, Xdom, ydom, cv=skf, method="predict_proba")[:, 1]
    domain_auc = float(roc_auc_score(ydom, oof))

    # ── Part 3: overlap region (new trials the classifier confuses with gold) ─
    new_oof = oof[len(Xg):]
    gold_oof = oof[:len(Xg)]
    overlap_mask = new_oof < 0.5                            # predicted gold-like
    gold_like_frac = float(overlap_mask.mean())
    overlap_est = int(round(gold_like_frac * n_new))
    # permissive upper estimate: within gold's central support (P(new) < gold 90th pct)
    perm_thr = float(np.percentile(gold_oof, 90))
    perm_frac = float((new_oof < perm_thr).mean())
    perm_est = int(round(perm_frac * n_new))
    # DECISIVE: completion rate WITHIN the overlap — is it gold-like (~0.28) or still ~0.81?
    new_status = S.set_index("nct_id").reindex(new_join["nct_id"])["overall_status"].to_numpy()
    overlap_pos = float((new_status[overlap_mask] == "COMPLETED").mean())

    # ── figures ───────────────────────────────────────────────────────────────
    def _save(name):
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = range(1998, 2027)
    ax.hist(gold_yr, bins=bins, density=True, alpha=0.6, color="#C44E52",
            label=f"gold Phase II (n={len(gold_yr)})")
    ax.hist(new_yr, bins=bins, density=True, alpha=0.5, color="#4C72B0",
            label=f"new pool (n={len(new_yr)})")
    ax.set_xlabel("registration year")
    ax.set_ylabel("density")
    ax.set_title("Phase II — registration-year shift: gold vs new pool")
    _save("phase2_expansion_year.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(gold_oof, bins=40, density=True, alpha=0.6, label="gold", color="#C44E52")
    ax.hist(new_oof, bins=40, density=True, alpha=0.5, label="new pool (sample)", color="#4C72B0")
    ax.axvline(0.5, ls="--", color="k", alpha=0.6, label="gold-like threshold 0.5")
    ax.set_xlabel("domain classifier P(new pool)")
    ax.set_ylabel("density")
    ax.set_title(f"Phase II domain separability (AUC={domain_auc:.3f})")
    _save("phase2_expansion_domain.png")

    # ── verdict ───────────────────────────────────────────────────────────────
    large_pool = n_new > 10000
    high_shift = domain_auc > 0.75
    large_overlap = overlap_est > 5000
    overlap_gold_like = abs(overlap_pos - gold_pos) < 0.12
    if large_pool and not high_shift:
        verdict = ("LOW-RISK, HIGH-VALUE — large new pool and low domain AUC (pools similar). "
                   "Proceed to Stage 2: add mechanical overall_status labels and retrain.")
    elif large_pool and high_shift and large_overlap and overlap_gold_like:
        verdict = (f"VIABLE WITH SHIFT CONTROL — large new pool, strong covariate shift "
                   f"(domain AUC {domain_auc:.3f}), BUT the ~{overlap_est:,}-trial overlap region is "
                   f"gold-like on BOTH covariates and label (overlap completion {overlap_pos:.2f} ≈ "
                   f"gold {gold_pos:.2f}). Stage 2 is worth trying via overlap-restriction (train on "
                   f"gold + the gold-like overlap slice) — NOT a naive bulk add (swings the prior to "
                   f"{new_pos:.2f} and reintroduces the Track-B shift). Recommend Stage 2 as a "
                   f"controlled experiment; verify it lifts gold-test PR-AUC through the promotion gate.")
    elif large_pool and high_shift and large_overlap and not overlap_gold_like:
        verdict = (f"MARGINAL — large overlap by covariates (~{overlap_est:,}) BUT it stays "
                   f"prior-shifted (overlap completion {overlap_pos:.2f} ≫ gold {gold_pos:.2f}). "
                   f"Importance-weighting corrects P(x), not this P(y) gap, so the addable-and-safe "
                   f"slice is smaller than the covariate overlap suggests. Low expected upside; "
                   f"pursue only as a scoped experiment, else document and go to writeup.")
    else:
        verdict = (f"DO NOT PURSUE (as a bulk add) — domain AUC {domain_auc:.3f} shows strong shift "
                   f"and the covariate-overlap region is small (~{overlap_est:,} trials). Mechanical "
                   f"expansion mostly imports routine {new_pos:.0%}-completion trials unlike the gold "
                   f"population; it would reintroduce the Track-B shift for little gold-like gain. "
                   f"Document as a finding and go to writeup.")

    # ── report ────────────────────────────────────────────────────────────────
    md = [
        "# Phase II label-expansion scope (Stage 1)\n",
        "Generated 2026-07-10. DIAGNOSTIC ONLY — no training, no matrix changed, no labels added.\n",
        "## 0. Label-definition linchpin (is the target mechanical?)\n",
        f"- Gold Phase II (membership `cto_gold ∩ cto_phase2`): **n={len(gp)}**, gold pos_rate **{gold_pos:.3f}**.",
        f"- Agreement `gold_y == (overall_status=='COMPLETED')`: **{label_agree:.4f}**.",
        "- → Gold is ~mechanical completion; the ~3% gap is COMPLETED-but-labeled-failure (faint "
        "efficacy signal). Mechanical `overall_status` labeling is label-clean to ~97% — same order "
        "as the known weak-vs-gold agreement. **Label axis is safe; the only risk is population shift.**",
        "",
        "## 1. Available expansion (counts)\n",
        f"- Routing set (matches CTO Phase II membership): `{sorted(ROUTE)}`.",
        f"- Interventional Phase II-routed trials (any status): **{len(intv):,}**.",
        f"- ... with terminal status ({sorted(TERMINAL)}): **{len(term):,}**.",
        f"- ... NOT already in gold Phase II → **NEW labelable: {n_new:,}**.",
        f"- Completion base rate — new pool **{new_pos:.3f}** vs gold **{gold_pos:.3f}**  "
        f"(**Δ={new_pos - gold_pos:+.3f}** — the Track-B population-prior signature).",
        "",
        "## 2. Covariate shift (the critical risk check)\n",
        f"- Registration-year median: gold **{int(gold_yr.median())}** vs new pool **{int(new_yr.median())}** "
        f"(see `phase2_expansion_year.png`).",
        f"- **Domain classifier AUC (gold vs new, {Xdom.shape[1]} features, 5-fold OOF): {domain_auc:.3f}**.",
        f"  - Comparable to the Track-B kill (whole-pop 0.912 / failure-class 0.940). "
        f"{'>0.75 → strong shift, easily separable → risky bulk add.' if high_shift else '~0.5-0.6 → pools similar → low-risk.'}",
        "",
        "## 3. Overlap region size\n",
        f"- New-pool trials the domain classifier confuses with gold (P(new)<0.5): "
        f"**{gold_like_frac:.1%}** of sample → **~{overlap_est:,}** of {n_new:,} (extrapolated).",
        f"- Permissive upper estimate (within gold's central support, P(new)<{perm_thr:.2f} = "
        f"gold 90th pct): **{perm_frac:.1%}** → **~{perm_est:,}**.",
        f"- **Completion rate WITHIN the overlap: {overlap_pos:.3f}** (vs gold {gold_pos:.3f}, "
        f"vs full new pool {new_pos:.3f}). "
        + ("Gold-like on BOTH covariates and label — the clean, addable subset."
           if abs(overlap_pos - gold_pos) < 0.12 else
           "Still prior-shifted even within the covariate overlap — IW on P(x) alone won't fix it."),
        "  (Sample-based extrapolation; see `phase2_expansion_domain.png`.)",
        "",
        "## 4. Temporal / leakage hygiene\n",
        f"- New-pool trials missing `study_first_posted_date` (no registration-time features): **{n_missing_date}**.",
        f"- Excluded as right-censored / non-terminal (RECRUITING/ACTIVE/UNKNOWN/etc.): **{n_excl:,}** — "
        f"never counted as completions. Breakdown: {excl.to_dict()}.",
        "- Features would be built by the audited 438-feature pipeline (registration-time only); "
        "`overall_status` is the LABEL, never a feature. No new leakage surface.",
        "",
        "## Verdict\n",
        f"**{verdict}**",
    ]
    OUT.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {OUT.relative_to(ROOT)} + 2 figures")


if __name__ == "__main__":
    main()
