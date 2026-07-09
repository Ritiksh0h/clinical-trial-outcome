#!/usr/bin/env python
"""
Dead-feature sweep — DIAGNOSTIC ONLY, changes nothing.

Motivated by the `accepts_healthy_volunteers` bug (boolean compared as string →
constant 0 for every trial). A feature that silently collapses to a constant
produces no error and passes every leakage test. This scan surfaces any sibling:
zero-variance / near-constant structured features, and dead (all-zero) TF-IDF terms.

Builds the current 430-feature matrix in-memory for the gold trials, per phase,
routed by CTO membership (which cto_phase{N}.parquet the nct_id appears in) — the
matrix Phase 2 will actually train on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"

NEAR_CONST_LO = 0.01  # binary pos_rate below this → near-constant
NEAR_CONST_HI = 0.99
STD_EPS = 1e-9


def _gold_phase_df(phase: int) -> pd.DataFrame:
    """Gold labels routed to `phase` by CTO membership, joined to studies cols."""
    from cto.features.build import _STUDIES_COLS
    from cto.features.leakage import drop_leakage_columns

    gold = pd.read_parquet(RAW / "cto_gold.parquet")  # nct_id, y
    member_ids = set(pd.read_parquet(RAW / f"cto_phase{phase}.parquet")["nct_id"])
    gold_phase = gold[gold["nct_id"].isin(member_ids)].copy()

    studies = drop_leakage_columns(
        pd.read_parquet(RAW / "aact_studies_snapshot.parquet"), warn=False
    )
    keep = [c for c in _STUDIES_COLS if c in studies.columns]
    return gold_phase.merge(studies[keep], on="nct_id", how="inner")


def _verdict(col: str, s: pd.Series) -> tuple[int, str, str]:
    """Return (nunique, stat_str, verdict)."""
    nun = s.nunique(dropna=False)
    vals = set(s.dropna().unique())
    is_binary = vals.issubset({0, 1}) and nun <= 2

    if nun <= 1:
        verdict = "EXPECTED-CONST" if col == "phase_clean" else "DEAD (constant)"
        return nun, f"pos_rate={s.mean():.4f}" if is_binary else f"const={s.iloc[0]}", verdict

    if is_binary:
        pr = float(s.mean())
        stat = f"pos_rate={pr:.4f}"
        if pr < NEAR_CONST_LO or pr > NEAR_CONST_HI:
            return nun, stat, "NEAR-CONSTANT"
        return nun, stat, "healthy"

    std = float(s.std())
    stat = f"std={std:.4g}"
    if std < STD_EPS:
        return nun, stat, "DEAD (zero-var)"
    return nun, stat, "healthy"


def main() -> None:
    print("# Dead-Feature Scan (diagnostic only)\n")
    all_flags: list[str] = []

    for phase in (1, 2, 3):
        df = _gold_phase_df(phase)
        from cto.features.build import build_features

        X = build_features(phase, "test", df=df)  # split=test → loads TF-IDF, never fits
        struct = [c for c in X.columns if not c.startswith("tfidf_")]
        tfidf = [c for c in X.columns if c.startswith("tfidf_")]

        print(f"\n## Phase {phase}  (n={len(X)}, {len(struct)} structured + {len(tfidf)} TF-IDF)\n")
        print(f"| {'feature':<28} | {'nunique':>7} | {'stat':<20} | verdict |")
        print(f"|{'-'*30}|{'-'*9}|{'-'*22}|{'-'*16}|")
        for col in struct:
            nun, stat, verdict = _verdict(col, X[col])
            mark = "" if verdict in ("healthy", "EXPECTED-CONST") else "  <<<"
            print(f"| {col:<28} | {nun:>7} | {stat:<20} | {verdict}{mark} |")
            if verdict.startswith(("DEAD", "NEAR")):
                all_flags.append(f"Phase {phase}: {col} → {verdict} ({stat})")

        # TF-IDF: dead = all-zero across this phase's sample
        dead_tfidf = [c for c in tfidf if float(X[c].abs().sum()) == 0.0]
        print(
            f"\nTF-IDF: {len(dead_tfidf)}/{len(tfidf)} columns all-zero (dead vocabulary) "
            f"for Phase {phase}"
        )

    print("\n## Summary of structured flags\n")
    if not all_flags:
        print(
            "No structured feature is dead or near-constant "
            "(phase_clean expected-constant per-phase is excluded)."
        )
    else:
        for f in all_flags:
            print(f"- {f}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
