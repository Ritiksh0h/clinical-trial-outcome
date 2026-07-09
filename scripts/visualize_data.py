#!/usr/bin/env python
"""
Phase 2 data-understanding visualizations.

Generates 4 charts to reports/figures/understanding/ explaining the
weak-vs-gold label divergence, the completion-not-efficacy finding, the
honest-vs-inflated model performance, and dataset sizes.

Run headlessly:
    python scripts/visualize_data.py

Charts 1/2/4 are computed from data/raw/*.parquet using the same _PHASE_MAP
routing as the pipeline (pure-phase subset). Chart 3 uses the fixed Phase 1
baseline metrics logged in MLflow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "figures" / "understanding"

PHASE_NAMES = {1: "Phase I", 2: "Phase II", 3: "Phase III"}
C_WEAK = "#DD8452"  # orange — over-optimistic weak labels
C_GOLD = "#4C72B0"  # blue — honest gold labels
C_STATUS = {
    "COMPLETED": "#55A868",  # green
    "TERMINATED": "#C44E52",  # red
    "WITHDRAWN": "#DD8452",  # orange
    "OTHER": "#8C8C8C",  # grey
}

# matches build.py exactly
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


def _gold_with_phase_status() -> pd.DataFrame:
    """Gold labels joined to AACT phase + overall_status, routed via _PHASE_MAP."""
    gold = pd.read_parquet(RAW / "cto_gold.parquet")
    studies = pd.read_parquet(
        RAW / "aact_studies_snapshot.parquet",
        columns=["nct_id", "phase", "overall_status"],
    )
    studies["phase_clean"] = studies["phase"].fillna("").str.lower().str.strip().map(_PHASE_MAP)
    return gold.merge(studies, on="nct_id", how="inner")


def _weak_pos_rates() -> dict[int, float]:
    return {p: pd.read_parquet(RAW / f"cto_phase{p}.parquet")["y"].mean() for p in (1, 2, 3)}


def _weak_sizes() -> dict[int, int]:
    return {p: len(pd.read_parquet(RAW / f"cto_phase{p}.parquet")) for p in (1, 2, 3)}


# ── Chart 1 — weak vs gold success rate ───────────────────────────────────────
def chart1_label_divergence(gold: pd.DataFrame) -> None:
    weak = _weak_pos_rates()
    gold_rate = {p: gold[gold["phase_clean"] == p]["y"].mean() for p in (1, 2, 3)}

    phases = [1, 2, 3]
    x = np.arange(len(phases))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 7))
    b1 = ax.bar(
        x - w / 2, [weak[p] for p in phases], w, label="Weak label success rate", color=C_WEAK
    )
    b2 = ax.bar(
        x + w / 2, [gold_rate[p] for p in phases], w, label="Gold label success rate", color=C_GOLD
    )

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.015,
                f"{h:.0%}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    # annotate the gap just above the shorter (gold) bar so it clears the legend
    for i, p in enumerate(phases):
        gap = weak[p] - gold_rate[p]
        ax.annotate(
            f"gap {gap:.0%}",
            xy=(i, gold_rate[p] + 0.055),
            ha="center",
            fontsize=11,
            color="#444",
            fontstyle="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_NAMES[p] for p in phases])
    ax.set_ylabel("Success rate (positive rate)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title(
        "Why the model broke: weak vs gold label success rates\ndiverge most in Phase I",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "chart1_weak_vs_gold_success_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Chart 2 — gold overall_status breakdown ───────────────────────────────────
def chart2_status_breakdown(gold: pd.DataFrame) -> None:
    def bucket(s: str) -> str:
        s = str(s).upper()
        return s if s in ("COMPLETED", "TERMINATED", "WITHDRAWN") else "OTHER"

    gold = gold.copy()
    gold["status_bucket"] = gold["overall_status"].map(bucket)

    order = ["COMPLETED", "TERMINATED", "WITHDRAWN", "OTHER"]
    phases = [1, 2, 3]

    # fraction per phase
    frac = {}
    for p in phases:
        sub = gold[gold["phase_clean"] == p]
        counts = sub["status_bucket"].value_counts(normalize=True)
        frac[p] = [counts.get(s, 0.0) for s in order]

    x = np.arange(len(phases))
    fig, ax = plt.subplots(figsize=(11, 7))
    bottom = np.zeros(len(phases))
    for i, status in enumerate(order):
        vals = np.array([frac[p][i] for p in phases])
        ax.bar(x, vals, 0.55, bottom=bottom, label=status, color=C_STATUS[status])
        # label segments >= 6%
        for j, v in enumerate(vals):
            if v >= 0.06:
                ax.text(
                    x[j],
                    bottom[j] + v / 2,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white",
                    fontweight="bold",
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_NAMES[p] for p in phases])
    ax.set_ylabel("Fraction of gold trials")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", frameon=True, fontsize=12)
    ax.set_title(
        "What 'failure' means: gold Phase I is mostly\nterminated/withdrawn trials",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "chart2_gold_status_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Chart 3 — honest vs inflated PR-AUC ───────────────────────────────────────
def chart3_performance_reality(gold: pd.DataFrame) -> None:
    # Fixed Phase 1 baseline metrics (logged in MLflow experiment cto_baseline)
    test_prauc = {1: 0.908, 2: 0.762, 3: 0.808}
    gold_prauc = {1: 0.358, 2: 0.514, 3: 0.700}
    # No-skill PR-AUC baseline = gold positive rate per phase
    no_skill = {p: gold[gold["phase_clean"] == p]["y"].mean() for p in (1, 2, 3)}

    phases = [1, 2, 3]
    x = np.arange(len(phases))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 7))
    b1 = ax.bar(
        x - w / 2,
        [test_prauc[p] for p in phases],
        w,
        label="Test PR-AUC (weak-label eval)",
        color=C_WEAK,
    )
    b2 = ax.bar(
        x + w / 2,
        [gold_prauc[p] for p in phases],
        w,
        label="Gold PR-AUC (honest eval)",
        color=C_GOLD,
    )

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )

    # no-skill baseline per phase group
    for i, p in enumerate(phases):
        ns = no_skill[p]
        ax.plot([i - w, i + w], [ns, ns], "k--", lw=2)
        ax.text(i + w + 0.02, ns, f"no-skill\n{ns:.0%}", va="center", fontsize=10, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_NAMES[p] for p in phases])
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True, fontsize=12)
    ax.set_title(
        "Honest vs inflated: gold evaluation reveals\nthe real performance",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "chart3_performance_reality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Chart 4 — dataset sizes ───────────────────────────────────────────────────
def chart4_dataset_sizes(gold: pd.DataFrame) -> None:
    weak = _weak_sizes()
    gold_n = {p: int((gold["phase_clean"] == p).sum()) for p in (1, 2, 3)}

    phases = [1, 2, 3]
    x = np.arange(len(phases))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 7))
    b1 = ax.bar(
        x - w / 2, [weak[p] for p in phases], w, label="Weak-label set (training)", color=C_WEAK
    )
    b2 = ax.bar(
        x + w / 2, [gold_n[p] for p in phases], w, label="Gold set (evaluation)", color=C_GOLD
    )

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 400,
                f"{int(h):,}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_NAMES[p] for p in phases])
    ax.set_ylabel("Number of trials")
    ax.legend(loc="upper right", frameon=True, fontsize=12)
    ax.set_title(
        "Dataset sizes: weak-label training set dwarfs\nthe gold evaluation set (~20×)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "chart4_dataset_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold = _gold_with_phase_status()

    # sanity check: routed gold pos_rates must match the confirmed findings
    for p, expected in [(1, 0.199), (2, 0.309), (3, 0.529)]:
        got = gold[gold["phase_clean"] == p]["y"].mean()
        assert abs(got - expected) < 0.01, f"Phase {p} pos_rate {got:.3f} != {expected}"

    chart1_label_divergence(gold)
    chart2_status_breakdown(gold)
    chart3_performance_reality(gold)
    chart4_dataset_sizes(gold)

    print(f"Saved 4 charts to {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
