"""
Track-B train/test contamination guard (audit Item 5).

Track B trains on weak+gold. 91.3% of gold TEST trials also appear in the weak phase
files, so training on all weak trials would leak the evaluation set into training. This
module is the ready-to-use guard: strip any weak trial whose nct_id is in the frozen
gold-test set before it can enter a weak-augmented training path.

Single source of truth: the gold-test nct_id list is PERSISTED by featurize_gold to
`data/processed/gold_test_nct_ids.json`. The guard reads that frozen file — it must NEVER
recompute the split independently, or two computations could drift and re-open the leak.

Not yet wired into any training path (whole-population Track B is deprioritized); this is
tested infrastructure so a future weak-augmentation cannot forget the exclusion.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

GOLD_TEST_IDS_PATH = Path(__file__).parents[3] / "data" / "processed" / "gold_test_nct_ids.json"


def filter_weak_excluding_gold_test(
    weak_df: pd.DataFrame, gold_test_nct_ids: Iterable[str]
) -> pd.DataFrame:
    """Return the subset of weak_df whose nct_id is NOT in the gold test set.

    Logs how many rows were removed. Does not mutate the input.
    """
    ids = set(gold_test_nct_ids)
    before = len(weak_df)
    safe = weak_df[~weak_df["nct_id"].isin(ids)].copy()
    removed = before - len(safe)
    logger.info(
        "contamination guard: removed %d of %d weak trials present in the gold test set "
        "(%.1f%%); %d remain.",
        removed,
        before,
        (100 * removed / before if before else 0.0),
        len(safe),
    )
    return safe


def save_gold_test_split(per_phase: dict, path: Path | str = GOLD_TEST_IDS_PATH) -> dict:
    """Persist the frozen gold-test split as {"phase1":[...], "phase2":[...],
    "phase3":[...], "all":[union]}. Called by featurize_gold with the per-phase test-set
    nct_ids taken from the SAME split objects that built the test matrices — the single
    source of truth. Returns the written structure.
    """
    obj = {f"phase{p}": sorted(set(per_phase.get(p, []))) for p in (1, 2, 3)}
    obj["all"] = sorted(set().union(*[set(v) for v in obj.values()]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=0)
    logger.info(
        "Persisted gold-test split to %s: phase1=%d phase2=%d phase3=%d all=%d",
        path,
        len(obj["phase1"]),
        len(obj["phase2"]),
        len(obj["phase3"]),
        len(obj["all"]),
    )
    return obj


def save_gold_test_nct_ids(nct_ids: Iterable[str], path: Path | str = GOLD_TEST_IDS_PATH) -> None:
    """Legacy flat-list writer (kept for back-compat). Prefer save_gold_test_split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(set(nct_ids)), f)
    logger.info("Persisted %d gold-test nct_ids to %s", len(set(nct_ids)), path)


def _load_raw(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — featurize_gold must persist the gold-test set "
            "(single source of truth) before the contamination guard can run. Never "
            "recompute the split independently."
        )
    with open(path) as f:
        return json.load(f)


def load_gold_test_nct_ids(
    path: Path | str = GOLD_TEST_IDS_PATH, phase: int | None = None
) -> list[str]:
    """Load frozen gold-test nct_ids. Returns the union ("all") by default, or a single
    phase's list if `phase` is given. Handles both the structured dict format and the
    legacy flat list. Single source of truth — never recompute the split elsewhere."""
    obj = _load_raw(path)
    if isinstance(obj, dict):
        return obj[f"phase{phase}"] if phase is not None else obj["all"]
    return obj  # legacy flat list


def load_gold_test_split(path: Path | str = GOLD_TEST_IDS_PATH) -> dict:
    """Load the full frozen structure {phase1, phase2, phase3, all}."""
    obj = _load_raw(path)
    if not isinstance(obj, dict):
        return {"all": obj}
    return obj
