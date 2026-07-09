"""
Build the sponsor-history interim table (DVC stage `sponsor_history`).

Amendment 3 HARD GATE (not a deferred manual check): after computing, sample >=500
TEST-set trials and RAISE if any has a sponsor_prior_trial_count that includes a trial
registered on/after its own study_first_posted_date. The gate runs on REAL test-set rows
and prints its evidence (per-trial computed vs actual prior counts).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RAW = Path(__file__).parents[3] / "data" / "raw"
_INTERIM = Path(__file__).parents[3] / "data" / "interim"

# Gold headline cutoffs (test = completed after this) — used only to pick TEST-set trials
# for the leakage gate, matching the featurize_gold split.
_VAL_CUTOFF = pd.Timestamp("2023-12-31")
_GATE_SAMPLE = 500


def _gold_test_nct_ids(studies: pd.DataFrame) -> list[str]:
    """Gold trials whose completion_date falls in the test window (2024+)."""
    gold = pd.read_parquet(_RAW / "cto_gold.parquet")
    comp = studies[["nct_id", "completion_date"]].copy()
    comp["completion_date"] = pd.to_datetime(comp["completion_date"], errors="coerce")
    g = gold.merge(comp, on="nct_id", how="inner")
    test = g[g["completion_date"] > _VAL_CUTOFF]
    return test["nct_id"].tolist()


def run() -> None:
    from cto.features.sponsor_history import (
        assert_no_future_leakage,
        assert_rate_outcome_known,
        compute_sponsor_history,
    )

    _INTERIM.mkdir(parents=True, exist_ok=True)
    studies = pd.read_parquet(_RAW / "aact_studies_snapshot.parquet")
    if "source" not in studies.columns:
        raise ValueError(
            "studies snapshot has no `source` column — sponsor identity is required. "
            "Re-pull studies with source (see aact_client.get_studies)."
        )

    logger.info("Computing sponsor history over %d studies…", len(studies))
    result = compute_sponsor_history(studies)

    # ── Amendment 3 hard gate on REAL test-set trials ─────────────────────────
    test_ids = _gold_test_nct_ids(studies)
    if len(test_ids) < _GATE_SAMPLE:
        logger.warning(
            "Only %d gold test-set trials available (< %d requested) — gating on all of them.",
            len(test_ids),
            _GATE_SAMPLE,
        )
    sample_ids = (
        pd.Series(test_ids).sample(min(_GATE_SAMPLE, len(test_ids)), random_state=42).tolist()
    )
    evidence = assert_no_future_leakage(
        result, studies, sample_ids, context="build_sponsor_history TEST-set gate"
    )
    # ITEM 3 gate: rates must use ONLY priors whose outcome was known before registration
    rate_evidence = assert_rate_outcome_known(
        result, studies, sample_ids, context="build_sponsor_history TEST-set gate"
    )

    checked = len(evidence)
    with_priors = [e for e in evidence if e["actual_prior"] > 0]
    print(f"\n=== Amendment 3 COUNT gate: PASSED on {checked} real TEST-set trials ===")
    print(
        f"(of {len(test_ids)} gold test-set trials; {len(with_priors)} of the checked have >0 priors)"
    )
    print("Sample evidence (computed prior_count vs actual strictly-earlier same-sponsor count):")
    for e in sorted(with_priors, key=lambda x: -x["actual_prior"])[:6]:
        print(
            f"  {e['nct_id']}: computed={e['computed_prior']:>4}  actual_strictly_prior={e['actual_prior']:>4}"
            f"  {'OK' if e['computed_prior'] <= e['actual_prior'] else 'LEAK'}"
        )

    print(
        f"\n=== ITEM 3 RATE gate: PASSED on {len(rate_evidence)} test trials with a known-outcome prior ==="
    )
    print("Stored completion_rate == honest (outcome-known-before) rate — sample:")
    for e in sorted(rate_evidence, key=lambda x: -x["n_known"])[:6]:
        print(
            f"  {e['nct_id']}: stored={e['stored_rate']:.4f}  honest={e['honest_rate']:.4f}  "
            f"n_known_priors={e['n_known']}"
        )

    # ── distribution of prior-trial counts (sparse vs dense sponsor history) ──
    pc = result["sponsor_prior_trial_count"]
    bins = {
        "0 priors": int((pc == 0).sum()),
        "1-2": int(((pc >= 1) & (pc <= 2)).sum()),
        "3-5": int(((pc >= 3) & (pc <= 5)).sum()),
        "6-20": int(((pc >= 6) & (pc <= 20)).sum()),
        "20+": int((pc > 20).sum()),
    }
    n = len(pc)
    print("\n=== sponsor_prior_trial_count distribution (all studies) ===")
    for k, v in bins.items():
        print(f"  {k:<9}: {v:>7,} ({100*v/n:.1f}%)")
    print(
        f"  established (>=5): {int(result['sponsor_is_established'].sum()):,} "
        f"({100*result['sponsor_is_established'].mean():.1f}%)"
    )

    out = _INTERIM / "sponsor_history.parquet"
    result.to_parquet(out, index=False)
    print(f"\nSaved {out} ({len(result):,} rows)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
