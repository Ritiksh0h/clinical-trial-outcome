"""Local Parquet snapshot of AACT data for fast offline feature building."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RAW_DIR = Path(__file__).parents[3] / "data" / "raw"
_SYNC_STATE = _RAW_DIR / "sync_state.json"


def get_last_sync() -> datetime | None:
    if not _SYNC_STATE.exists():
        return None
    with open(_SYNC_STATE) as f:
        data = json.load(f)
    ts = data.get("last_sync")
    return datetime.fromisoformat(ts) if ts else None


def _save_sync_state(ts: datetime) -> None:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SYNC_STATE, "w") as f:
        json.dump({"last_sync": ts.isoformat()}, f)


def build_mirror(since: datetime | None = None) -> None:
    """
    Pull AACT tables and save as Parquet snapshots.
    Incremental when since is provided; full pull when since is None.
    """
    from cto.data.aact_client import (
        get_calculated_values,
        get_conditions,
        get_countries,
        get_designs,
        get_eligibilities,
        get_interventions,
        get_sponsors,
        get_studies,
    )

    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building AACT mirror (since=%s)…", since)
    studies = get_studies(since=since)

    # On an incremental pull, 0 rows means "nothing changed" — preserve the
    # existing snapshot rather than overwriting it with an empty file.
    snapshot_path = _RAW_DIR / "aact_studies_snapshot.parquet"
    if since is not None and len(studies) == 0:
        logger.info("Incremental pull: no new studies since last sync — keeping existing snapshot.")
        return

    studies.to_parquet(snapshot_path, index=False)
    logger.info("  studies: %d rows", len(studies))

    nct_ids = studies["nct_id"].tolist()
    if not nct_ids:
        logger.warning("No NCT IDs returned from studies — mirror tables will be empty.")
        return

    tables = {
        "designs": get_designs,
        "eligibilities": get_eligibilities,
        "sponsors": get_sponsors,
        "conditions": get_conditions,
        "interventions": get_interventions,
        "calculated_values": get_calculated_values,
        "countries": get_countries,
    }
    for name, fn in tables.items():
        df = fn(nct_ids)
        # Never overwrite a snapshot with an empty DataFrame (CLAUDE.md mirror guard).
        if len(df) == 0:
            logger.warning(
                "  %s: 0 rows returned — keeping existing snapshot, skipping write.", name
            )
            continue
        df.to_parquet(_RAW_DIR / f"aact_{name}_snapshot.parquet", index=False)
        logger.info("  %s: %d rows", name, len(df))

    _save_sync_state(datetime.now(UTC))
    logger.info("Mirror complete. Sync state saved.")


def load_mirror(table: str) -> pd.DataFrame:
    """Load a saved AACT snapshot parquet."""
    path = _RAW_DIR / f"aact_{table}_snapshot.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Mirror file not found: {path}. Run `python -m cto.pipelines.ingest` first."
        )
    return pd.read_parquet(path)
