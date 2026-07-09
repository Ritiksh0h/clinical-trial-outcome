"""
EDA for CTO Phase 0.
Runs headlessly; saves all figures to reports/figures/.
Use # %% markers for IDE cell support.
"""

from __future__ import annotations

# %% Imports
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # headless

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

_RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
_FIG_DIR = Path(__file__).parents[1] / "reports" / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────


def _load_phase(n: int) -> pd.DataFrame | None:
    p = _RAW_DIR / f"cto_phase{n}.parquet"
    if not p.exists():
        logger.warning("Missing %s — run `dvc repro ingest` first", p.name)
        return None
    return pd.read_parquet(p)


def _load_studies() -> pd.DataFrame | None:
    p = _RAW_DIR / "aact_studies_snapshot.parquet"
    if not p.exists():
        logger.warning("Missing aact_studies_snapshot.parquet — AACT mirror not built")
        return None
    return pd.read_parquet(p)


def _save(fig: plt.Figure, name: str) -> None:
    path = _FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)


# %% Section 1 — Label distribution per phase


def section_label_distribution() -> None:
    logger.info("\n=== Section 1: Label distribution ===")
    rows = []
    for n in [1, 2, 3]:
        df = _load_phase(n)
        if df is None or "y" not in df.columns:
            continue
        success_rate = df["y"].mean() * 100
        rows.append(
            {
                "phase": f"Phase {n}",
                "success_rate": success_rate,
                "success": df["y"].sum(),
                "failure": (df["y"] == 0).sum(),
            }
        )
        logger.info("  Phase %d: %.1f%% success (%d total)", n, success_rate, len(df))

    if not rows:
        logger.warning("No phase data available — skipping label distribution plots")
        return

    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(summary["phase"], summary["success_rate"], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].axhline(50, color="red", linestyle="--", alpha=0.5, label="50% baseline")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Success Rate per Phase")
    axes[0].legend()

    x = range(len(summary))
    axes[1].bar(x, summary["success"], label="Success", color="#55A868")
    axes[1].bar(x, summary["failure"], bottom=summary["success"], label="Failure", color="#C44E52")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(summary["phase"])
    axes[1].set_ylabel("Count")
    axes[1].set_title("Label Counts per Phase")
    axes[1].legend()

    fig.suptitle("CTO Label Distribution")
    _save(fig, "label_distribution.png")


# %% Section 2 — pred_proba distribution


def section_pred_proba_distribution() -> None:
    logger.info("\n=== Section 2: pred_proba distribution ===")
    # Load raw CTO CSVs before the binary-label transform
    from datasets import load_dataset

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    configs = [
        ("phase1_CTO_preds", "Phase 1", axes[0]),
        ("phase2_CTO_preds", "Phase 2", axes[1]),
        ("phase3_CTO_preds", "Phase 3", axes[2]),
    ]
    for config_name, label, ax in configs:
        try:
            ds = load_dataset("chufangao/CTO", config_name, split="test", trust_remote_code=False)
            df = ds.to_pandas()
            logger.info(
                "  %s: %d rows, %d unique nct_ids, pred_proba null=%.1f%%",
                config_name,
                len(df),
                df["nct_id"].nunique() if "nct_id" in df.columns else -1,
                df["pred_proba"].isna().mean() * 100 if "pred_proba" in df.columns else 0,
            )
            if "pred_proba" in df.columns:
                ax.hist(df["pred_proba"].dropna(), bins=50, color="#4C72B0", edgecolor="white")
            ax.set_title(label)
            ax.set_xlabel("pred_proba")
            ax.set_ylabel("Count")
        except Exception as exc:
            logger.warning("  Could not load %s: %s", config_name, exc)
            ax.set_title(f"{label} (unavailable)")

    fig.suptitle("CTO pred_proba Distribution per Phase")
    _save(fig, "pred_proba_distribution.png")


# %% Section 3 — AACT join rate


def section_join_rate() -> None:
    logger.info("\n=== Section 3: AACT join rate ===")
    studies = _load_studies()
    if studies is None:
        logger.warning("Skipping join rate — AACT snapshot not available")
        # Write a placeholder figure
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "AACT mirror not built.\nRun dvc repro ingest with credentials.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.axis("off")
        _save(fig, "join_rate.png")
        return

    rows = []
    for n in [1, 2, 3]:
        df_labels = _load_phase(n)
        if df_labels is None:
            continue
        joined = df_labels.merge(studies[["nct_id"]], on="nct_id", how="inner")
        join_rate = 100 * len(joined) / max(len(df_labels), 1)
        rows.append(
            {
                "phase": f"Phase {n}",
                "cto_rows": len(df_labels),
                "joined": len(joined),
                "join_rate": join_rate,
            }
        )
        flag = " ⚠ LOW" if join_rate < 50 else ""
        logger.info(
            "  Phase %d: %d CTO → %d joined (%.1f%%){%s}",
            n,
            len(df_labels),
            len(joined),
            join_rate,
            flag,
        )

    if not rows:
        return
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E52" if r < 50 else "#55A868" for r in summary["join_rate"]]
    ax.bar(summary["phase"], summary["join_rate"], color=colors)
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_ylabel("Join rate (%)")
    ax.set_title("CTO → AACT Inner Join Rate per Phase")
    ax.legend()
    _save(fig, "join_rate.png")


# %% Section 4 — Registration-time feature distributions


def section_feature_distributions() -> None:
    logger.info("\n=== Section 4: Feature distributions ===")
    studies = _load_studies()
    if studies is None or studies.empty:
        logger.warning("Skipping feature distributions — AACT snapshot not available")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "AACT mirror not built.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.axis("off")
        _save(fig, "feature_distributions.png")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Enrollment histogram (log scale)
    ax = axes[0, 0]
    enroll = (
        studies.loc[studies.get("enrollment_type", pd.Series()) == "ANTICIPATED", "enrollment"]
        if "enrollment_type" in studies.columns
        else studies.get("enrollment", pd.Series())
    )
    enroll = pd.to_numeric(enroll, errors="coerce").dropna()
    if len(enroll):
        ax.hist(enroll.clip(upper=enroll.quantile(0.99)), bins=60, log=True, color="#4C72B0")
        ax.set_xlabel("Enrollment (ANTICIPATED)")
        ax.set_ylabel("Count (log)")
        ax.set_title("Enrollment Distribution")

    # study_first_posted_date histogram
    ax = axes[0, 1]
    if "study_first_posted_date" in studies.columns:
        dates = pd.to_datetime(studies["study_first_posted_date"], errors="coerce")
        years = dates.dt.year.dropna().astype(int)
        years[years.between(2000, 2024)].hist(bins=25, ax=ax, color="#DD8452")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.set_title("Study Registration Year")

    # source_class bar chart
    ax = axes[1, 0]
    if "source_class" in studies.columns:
        vc = studies["source_class"].fillna("UNKNOWN").value_counts().head(8)
        vc.plot.barh(ax=ax, color="#55A868")
        ax.set_title("Sponsor Class (source_class)")
        ax.set_xlabel("Count")

    # overall_status bar chart
    ax = axes[1, 1]
    if "overall_status" in studies.columns:
        vc = studies["overall_status"].fillna("UNKNOWN").value_counts().head(8)
        vc.plot.barh(ax=ax, color="#8172B2")
        ax.set_title("Overall Status")
        ax.set_xlabel("Count")

    fig.suptitle("Registration-Time Feature Distributions (AACT)")
    fig.tight_layout()
    _save(fig, "feature_distributions.png")


# %% Section 5 — Temporal coverage


def section_temporal_coverage() -> None:
    logger.info("\n=== Section 5: Temporal coverage ===")
    studies = _load_studies()
    if studies is None or studies.empty:
        logger.warning("Skipping temporal coverage — AACT snapshot not available")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(
            0.5,
            0.5,
            "AACT mirror not built.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.axis("off")
        _save(fig, "temporal_coverage.png")
        return

    if "study_first_posted_date" not in studies.columns:
        return
    studies = studies.copy()
    studies["reg_year"] = pd.to_datetime(
        studies["study_first_posted_date"], errors="coerce"
    ).dt.year

    _PHASE_MAP = {
        "phase 1": 1,
        "phase1": 1,
        "phase i": 1,
        "phase 2": 2,
        "phase2": 2,
        "phase ii": 2,
        "phase 3": 3,
        "phase3": 3,
        "phase iii": 3,
    }
    if "phase" in studies.columns:
        studies["phase_clean"] = studies["phase"].fillna("").str.lower().str.strip().map(_PHASE_MAP)

    fig, ax = plt.subplots(figsize=(14, 6))
    for phase_n, color in [(1, "#4C72B0"), (2, "#DD8452"), (3, "#55A868")]:
        sub = (
            studies[studies.get("phase_clean") == phase_n]
            if "phase_clean" in studies.columns
            else studies
        )
        counts = sub["reg_year"].value_counts().sort_index()
        counts = counts[(counts.index >= 2000) & (counts.index <= 2024)]
        ax.plot(
            counts.index,
            counts.values,
            label=f"Phase {phase_n}",
            color=color,
            marker="o",
            markersize=3,
        )

    ax.axvline(
        2022, color="red", linestyle="--", alpha=0.7, label="Proposed train/test cutoff (2022)"
    )
    ax.set_xlabel("Registration Year")
    ax.set_ylabel("Number of Trials")
    ax.set_title("Trial Registrations per Year per Phase (2000–2024)")
    ax.legend()
    _save(fig, "temporal_coverage.png")


# %% Section 6 — Leakage surface audit


def section_leakage_audit() -> None:
    logger.info("\n=== Section 6: Leakage surface audit ===")
    from cto.features.leakage import LEAKAGE_BLOCKLIST
    from datasets import load_dataset

    configs = [
        ("phase1_CTO_preds", "Phase 1"),
        ("phase2_CTO_preds", "Phase 2"),
        ("phase3_CTO_preds", "Phase 3"),
        ("human_labels", "Gold"),
    ]
    lines: list[str] = ["CTO Leakage Surface Audit", "=" * 60]
    for config_name, label in configs:
        try:
            ds = load_dataset("chufangao/CTO", config_name, split="test", trust_remote_code=False)
            df = ds.to_pandas()
            leaked = sorted(c for c in df.columns if c in LEAKAGE_BLOCKLIST)
            safe = sorted(c for c in df.columns if c not in LEAKAGE_BLOCKLIST)
            lines.append(f"\n{label} ({config_name})")
            lines.append(f"  Total columns: {len(df.columns)}")
            lines.append(f"  BLOCKLISTED ({len(leaked)}): {leaked}")
            lines.append(f"  Safe ({len(safe)}): {safe}")
            logger.info("  %s: %d/%d columns blocklisted", label, len(leaked), len(df.columns))
        except Exception as exc:
            lines.append(f"\n{label}: ERROR — {exc}")

    audit_path = _FIG_DIR / "leakage_audit.txt"
    audit_path.write_text("\n".join(lines))
    logger.info("Saved leakage_audit.txt")


# %% Section 7 — Missing value heatmap (Phase 3)


def section_missing_values_phase3() -> None:
    logger.info("\n=== Section 7: Missing values (Phase 3) ===")
    studies = _load_studies()
    labels = _load_phase(3)
    if studies is None or studies.empty or labels is None:
        logger.warning("Skipping missing-value heatmap — data not available")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "Data not available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.axis("off")
        _save(fig, "missing_values_phase3.png")
        return

    from cto.features.build import build_raw_joined

    try:
        df = build_raw_joined(phase=3)
    except Exception as exc:
        logger.warning("build_raw_joined failed: %s — using direct join for heatmap", exc)
        df = labels.merge(studies, on="nct_id", how="inner")

    null_rates = df.isnull().mean().rename("null_rate").reset_index()
    null_rates.columns = ["column", "null_rate"]
    null_rates = null_rates.sort_values("null_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(null_rates) * 0.3)))
    colors = ["#C44E52" if r > 0.5 else "#4C72B0" for r in null_rates["null_rate"]]
    ax.barh(null_rates["column"], null_rates["null_rate"] * 100, color=colors)
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label=">50% flag")
    ax.set_xlabel("Null rate (%)")
    ax.set_title("Null Rate per Column — Phase 3 Joined DataFrame")
    ax.legend()
    fig.tight_layout()
    _save(fig, "missing_values_phase3.png")


# %% Main


if __name__ == "__main__":
    logger.info("Running EDA — outputs to reports/figures/")
    section_label_distribution()
    section_pred_proba_distribution()
    section_join_rate()
    section_feature_distributions()
    section_temporal_coverage()
    section_leakage_audit()
    section_missing_values_phase3()
    logger.info("\nEDA complete. Figures in %s", _FIG_DIR)
