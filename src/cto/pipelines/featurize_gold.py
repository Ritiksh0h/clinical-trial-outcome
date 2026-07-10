"""
Gold-label featurize pipeline (Phase 2 Step 4).

AUTHORITATIVE gold test set. featurize_gold is the SOLE computer of the gold temporal
split. The SAME split object per phase is used for two things: (1) building the
train/val/test feature matrices, and (2) freezing the test-set nct_ids to
data/processed/gold_test_nct_ids.json. Do NOT recompute the gold split anywhere else —
read the frozen file via cto.features.contamination_guard.load_gold_test_nct_ids().

Phase routing: CTO MEMBERSHIP — a trial is in phase N iff its nct_id appears in
cto_phaseN.parquet; combo trials appear in both their phases. Chosen over _PHASE_MAP:
  - consistent with all prior analysis (audit positive counts → walk-forward gate,
    covariate-shift, label-vs-population) which used membership; switching invalidates them;
  - _PHASE_MAP drops ~40% of Phase I gold (PHASE1/PHASE2 combo + EARLY_PHASE1) — Phase I is
    already the scarcest phase, don't discard it;
  - combo trials in two phases are HARMLESS: the temporal split is a pure function of
    completion_date with shared cutoffs, so a combo trial lands in the SAME split in both
    phases (pre_phase2_audit Item 1) → no cross-phase leakage. The thing _PHASE_MAP "fixes"
    is a non-problem.

Split cutoffs (gold): train ≤2022-12-31 / val 2023 / test 2024+ (params.yaml split.gold_*).
TF-IDF is NOT refit — build_features loads the Phase 1 vectorizer.
Schema 2.1.0, 438 features (facilities/num_countries/is_multinational dropped as soft leakage).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[3]
_RAW = _ROOT / "data" / "raw"
_PROCESSED = _ROOT / "data" / "processed"
_PARAMS = _ROOT / "params.yaml"

EXPECTED_TEST_COUNTS = {1: 410, 2: 555, 3: 275, "all": 1073}


def run() -> None:
    from cto.features.build import _STUDIES_COLS, build_features
    from cto.features.contamination_guard import save_gold_test_split
    from cto.features.leakage import drop_leakage_columns
    from cto.features.split import assert_temporal_integrity, make_temporal_splits
    from cto.features.text import _TFIDF_PATH

    if not _TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {_TFIDF_PATH}. Run Phase 1 featurize first "
            "(the gold pipeline must reuse the existing vocabulary, never refit)."
        )

    params = yaml.safe_load(open(_PARAMS))
    gold_train = params["split"]["gold_train_cutoff"]
    gold_val = params["split"]["gold_val_cutoff"]

    gold = pd.read_parquet(_RAW / "cto_gold.parquet")  # nct_id, y
    studies = drop_leakage_columns(
        pd.read_parquet(_RAW / "aact_studies_snapshot.parquet"), warn=False
    )
    studies_k = studies[[c for c in _STUDIES_COLS if c in studies.columns]]

    _PROCESSED.mkdir(parents=True, exist_ok=True)
    frozen_test: dict[int, list[str]] = {}
    summary = []

    for phase in (1, 2, 3):
        member = set(
            pd.read_parquet(_RAW / f"cto_phase{phase}.parquet")["nct_id"]
        )  # CTO membership
        gp = gold[gold["nct_id"].isin(member)].merge(studies_k, on="nct_id", how="inner")

        # ── ONE split object per phase — used for BOTH the matrices and the frozen list ──
        splits = make_temporal_splits(
            gp, date_col="completion_date", train_cutoff=gold_train, val_cutoff=gold_val
        )
        assert_temporal_integrity(splits["train"], splits["val"], splits["test"])
        frozen_test[phase] = splits["test"]["nct_id"].tolist()  # (2nd use — same object)

        for name in ("train", "val", "test"):
            sdf = splits[name]  # (1st use — same object)
            y = sdf["y"].reset_index(drop=True)
            X = build_features(phase, name, df=sdf)
            if X.shape[1] <= 427:
                raise ValueError(
                    f"gold matrix has {X.shape[1]} features (≤427) — sponsor/TA join missing"
                )
            X.to_parquet(_PROCESSED / f"features_gold_phase{phase}_{name}.parquet", index=False)
            y.to_frame("y").to_parquet(
                _PROCESSED / f"labels_gold_phase{phase}_{name}.parquet", index=False
            )
            summary.append((phase, name, X.shape[0], X.shape[1], int(y.sum()), int((y == 0).sum())))

    # ── Freeze the gold test set from the SAME split objects (single source of truth) ──
    frozen = save_gold_test_split(frozen_test)

    # ── report ────────────────────────────────────────────────────────────────
    print("\n=== gold matrices (features_gold_phase{p}_{split}.parquet) ===")
    print(f"{'phase':>5} {'split':>6} {'rows':>6} {'feats':>6} {'pos':>5} {'neg':>6}")
    for phase, name, n, feats, pos, neg in summary:
        print(f"{phase:>5} {name:>6} {n:>6} {feats:>6} {pos:>5} {neg:>6}")

    print("\n=== frozen gold test set (data/processed/gold_test_nct_ids.json) ===")
    ok = True
    for phase in (1, 2, 3):
        got, exp = len(frozen[f"phase{phase}"]), EXPECTED_TEST_COUNTS[phase]
        mark = "OK" if got == exp else f"** expected {exp}"
        ok &= got == exp
        print(f"  phase{phase}: {got}  {mark}")
    got_all = len(frozen["all"])
    ok &= got_all == EXPECTED_TEST_COUNTS["all"]
    print(
        f"  all(union): {got_all}  {'OK' if got_all == EXPECTED_TEST_COUNTS['all'] else '** expected ' + str(EXPECTED_TEST_COUNTS['all'])}"
    )
    print(f"\nfrozen counts match expected (membership): {ok}")
    print(f"schema_version: {params['features']['schema_version']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
