"""
Build the indication (therapeutic-area) history interim table (DVC stage `indication_history`).

HARD GATE (same as sponsor Amendment 3): after computing, sample >=500 real TEST-set trials
and run BOTH the COUNT gate (no future-registered prior counted) and the RATE gate (no prior
with completion_date >= the current registration counted in a rate). RAISE on violation.
Prints evidence including count-vs-rate prior numbers, showing the two windows differ.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RAW = Path(__file__).parents[3] / "data" / "raw"
_INTERIM = Path(__file__).parents[3] / "data" / "interim"
_VAL_CUTOFF = pd.Timestamp("2023-12-31")
_GATE_SAMPLE = 500

_PHASE_MAP = {
    "1": 1, "phase 1": 1, "phase1": 1, "phase i": 1,
    "2": 2, "phase 2": 2, "phase2": 2, "phase ii": 2,
    "3": 3, "phase 3": 3, "phase3": 3, "phase iii": 3,
}


def _gold_test_nct_ids(studies: pd.DataFrame) -> list[str]:
    gold = pd.read_parquet(_RAW / "cto_gold.parquet")
    comp = studies[["nct_id", "completion_date"]].copy()
    comp["completion_date"] = pd.to_datetime(comp["completion_date"], errors="coerce")
    g = gold.merge(comp, on="nct_id", how="inner")
    return g[g["completion_date"] > _VAL_CUTOFF]["nct_id"].tolist()


def run() -> None:
    from cto.features.indication_history import (
        TA_CODE,
        assert_no_future_leakage_ta,
        assert_rate_outcome_known_ta,
        assign_therapeutic_area,
        compute_indication_history,
    )

    _INTERIM.mkdir(parents=True, exist_ok=True)
    studies = pd.read_parquet(_RAW / "aact_studies_snapshot.parquet")
    conditions = pd.read_parquet(_RAW / "aact_conditions_snapshot.parquet")

    logger.info("Computing indication history over %d studies…", len(studies))
    result = compute_indication_history(studies, conditions)

    # ── hard gate on REAL test-set trials ─────────────────────────────────────
    test_ids = _gold_test_nct_ids(studies)
    sample_ids = pd.Series(test_ids).sample(min(_GATE_SAMPLE, len(test_ids)), random_state=42).tolist()
    count_ev = assert_no_future_leakage_ta(result, studies, conditions, sample_ids, context="build TEST-set gate")
    rate_ev = assert_rate_outcome_known_ta(result, studies, conditions, sample_ids, context="build TEST-set gate")
    cnt_of = dict(zip(result["nct_id"], result["ta_prior_trial_count"], strict=False))

    print(f"\n=== COUNT gate: PASSED on {len(count_ev)} real TEST-set trials ===")
    for e in sorted(count_ev, key=lambda x: -x["actual_prior"])[:5]:
        print(f"  {e['nct_id']}: computed={e['computed_prior']}  actual_reg_prior={e['actual_prior']}"
              f"  {'OK' if e['computed_prior'] <= e['actual_prior'] else 'LEAK'}")
    print(f"\n=== RATE gate: PASSED on {len(rate_ev)} test trials with a known-outcome prior ===")
    print("Stored rate == honest (outcome-known) rate; count vs rate-window prior numbers differ:")
    for e in sorted(rate_ev, key=lambda x: -x["n_known"])[:5]:
        print(f"  {e['nct_id']}: stored={e['stored_rate']:.4f} honest={e['honest_rate']:.4f}  "
              f"count_prior={cnt_of.get(e['nct_id'])}  rate_known_prior={e['n_known']}")

    # ── OTHER-bucket fraction + full bucket distribution ──────────────────────
    ta = assign_therapeutic_area(studies["nct_id"], conditions)
    code2name = {v: k for k, v in TA_CODE.items()}
    dist = result["ta_bucket"].map(code2name).value_counts()
    n = len(result)
    print(f"\n=== Therapeutic-area bucket distribution (all {n:,} studies) ===")
    for name, c in dist.items():
        print(f"  {name:<14}: {c:>7,} ({100*c/n:.1f}%)")
    print(f"  >>> OTHER (unmapped) fraction: {100*(ta=='OTHER').mean():.1f}%")

    # ── per-phase rate non-fill fraction ──────────────────────────────────────
    ph = studies["phase"].fillna("").astype(str).str.lower().str.strip().map(_PHASE_MAP)
    phase_of = dict(zip(studies["nct_id"], ph, strict=False))
    gold_ids = set(pd.read_parquet(_RAW / "cto_gold.parquet")["nct_id"])
    R = pd.to_datetime(studies["study_first_posted_date"], errors="coerce").to_numpy()
    C = pd.to_datetime(studies["completion_date"], errors="coerce").to_numpy()
    ta_arr = ta.to_numpy()
    reg_of = dict(zip(studies["nct_id"], R, strict=False))
    print("\n=== rate non-fill fraction on GOLD trials, per phase ===")
    import numpy as np
    for target in (1, 2, 3):
        ids = [nid for nid in gold_ids if phase_of.get(nid) == target]
        nonfill = 0
        ta_by = {nid: t for nid, t in zip(studies["nct_id"], ta_arr, strict=False)}
        for nid in ids:
            r = reg_of.get(nid)
            if pd.isna(r):
                continue
            if ((ta_arr == ta_by[nid]) & (C < np.datetime64(r))).sum() > 0:
                nonfill += 1
        if ids:
            print(f"  Phase {target}: {nonfill}/{len(ids)} gold trials have a real (non-filled) rate ({100*nonfill/len(ids):.1f}%)")

    out = _INTERIM / "indication_history.parquet"
    result.to_parquet(out, index=False)
    print(f"\nSaved {out} ({len(result):,} rows)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
