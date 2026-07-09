"""
Ingest pipeline — called by `dvc repro ingest`.
1. Downloads CTO label files from HuggingFace.
2. Builds/refreshes the AACT mirror (incremental on re-runs, full on first run).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PARAMS_PATH = Path(__file__).parents[3] / "params.yaml"


def _load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def run() -> None:
    # ── Step 1: CTO labels ────────────────────────────────────────────────────
    logger.info("=== Ingesting CTO labels from HuggingFace ===")
    from cto.data.cto_labels import load_all

    data = load_all(save=True)
    for name, df in data.items():
        logger.info("  %s: %d rows", name, len(df))

    # ── Step 2: AACT mirror ───────────────────────────────────────────────────
    from cto.common.settings import settings

    if not settings.aact_user or not settings.aact_password:
        logger.error(
            "AACT credentials not set — skipping AACT mirror build.\n"
            "Set AACT_USER and AACT_PASSWORD in your .env file and re-run.\n"
            "Register free at https://aact.ctti-clinicaltrials.org/users/sign_up"
        )
        # Write an empty studies snapshot so dvc.yaml outputs are satisfied
        import pandas as pd

        empty_path = Path(__file__).parents[3] / "data" / "raw" / "aact_studies_snapshot.parquet"
        pd.DataFrame(
            columns=[
                "nct_id",
                "phase",
                "study_type",
                "overall_status",
                "enrollment",
                "enrollment_type",
                "number_of_arms",
                "number_of_groups",
                "source_class",
                "study_first_posted_date",
                "primary_completion_date",
                "completion_date",
                "last_update_posted_date",
            ]
        ).to_parquet(empty_path, index=False)
        logger.info("  Wrote empty AACT snapshot placeholder.")
        return

    logger.info("=== Building AACT mirror ===")
    from cto.data.mirror import build_mirror, get_last_sync

    since = get_last_sync()
    if since:
        logger.info("  Incremental pull since %s", since.isoformat())
    else:
        logger.info("  First run — full pull (may take several minutes)")
    build_mirror(since=since)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
