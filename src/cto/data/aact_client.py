"""
PostgreSQL client for AACT with incremental-pull support.
Only registration-time columns are ever queried — post-hoc columns are never selected.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

logger = logging.getLogger(__name__)

_AACT_YAML = Path(__file__).parents[3] / "config" / "aact.yaml"
_AACT_SIGNUP_URL = "https://aact.ctti-clinicaltrials.org/users/sign_up"
_CHUNK_SIZE = 5_000


def _load_aact_config() -> dict[str, Any]:
    with open(_AACT_YAML) as f:
        return yaml.safe_load(f)


def get_engine() -> Engine:
    """Build a SQLAlchemy engine for AACT. Reads credentials from settings."""
    from cto.common.settings import settings

    if not settings.aact_user or not settings.aact_password:
        raise OSError(
            "AACT credentials not set. "
            f"Register for a free account at {_AACT_SIGNUP_URL}, "
            "then set AACT_USER and AACT_PASSWORD in your .env file."
        )

    cfg = _load_aact_config()
    # NEVER replace this with an f-string URL. If the password contains @, /, :, or
    # other special characters, f-string interpolation silently embeds them into the
    # hostname, producing a DNS error that points at the wrong place entirely.
    # URL.create() percent-encodes credentials before building the connection string.
    url = URL.create(
        "postgresql+psycopg2",
        username=settings.aact_user,
        password=settings.aact_password,
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["dbname"],
        query={"sslmode": cfg["sslmode"]},
    )
    return create_engine(url, pool_pre_ping=True)


def get_studies(since: datetime | None = None) -> pd.DataFrame:
    """
    Fetch registration-time study metadata.
    If since is None, pulls all ~400k studies (warn: slow).
    If since is provided, pulls only rows updated after that datetime.
    Never selects post-hoc columns.
    """
    if since is None:
        logger.warning(
            "Pulling ALL studies from AACT (~400k rows). "
            "This may take several minutes. Set AACT_LAST_SYNC to skip."
        )

    cfg = _load_aact_config()
    schema = cfg["schema"]
    cols = (
        "nct_id, phase, study_type, overall_status, enrollment, enrollment_type, "
        "number_of_arms, number_of_groups, source, source_class, "
        "study_first_posted_date, primary_completion_date, completion_date, "
        "last_update_posted_date"
    )
    base_q = f"SELECT {cols} FROM {schema}.studies"  # noqa: S608
    if since is not None:
        ts = since.strftime("%Y-%m-%d %H:%M:%S")
        q = f"{base_q} WHERE last_update_posted_date > '{ts}'"
    else:
        q = base_q

    try:
        engine = get_engine()
        df = pd.read_sql(text(q), engine)
    except Exception as exc:
        _handle_connection_error(exc)
        raise

    logger.info("get_studies: %d rows fetched", len(df))
    return df


def _chunked_query(table: str, cols: str, nct_ids: list[str]) -> pd.DataFrame:
    """Run a query in chunks to avoid PostgreSQL query size limits."""
    cfg = _load_aact_config()
    schema = cfg["schema"]
    engine = get_engine()
    frames: list[pd.DataFrame] = []
    for i in range(0, len(nct_ids), _CHUNK_SIZE):
        chunk = nct_ids[i : i + _CHUNK_SIZE]
        placeholders = ", ".join(f"'{nid}'" for nid in chunk)
        q = f"SELECT {cols} FROM {schema}.{table} WHERE nct_id IN ({placeholders})"  # noqa: S608
        frames.append(pd.read_sql(text(q), engine))
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols.split(", "))
    )


def get_designs(nct_ids: list[str]) -> pd.DataFrame:
    cols = "nct_id, allocation, intervention_model, masking, masking_description, primary_purpose"
    return _chunked_query("designs", cols, nct_ids)


def get_eligibilities(nct_ids: list[str]) -> pd.DataFrame:
    cols = "nct_id, gender, minimum_age, maximum_age, healthy_volunteers, criteria"
    return _chunked_query("eligibilities", cols, nct_ids)


def get_sponsors(nct_ids: list[str]) -> pd.DataFrame:
    cols = "nct_id, agency_class, lead_or_collaborator"
    return _chunked_query("sponsors", cols, nct_ids)


def get_conditions(nct_ids: list[str]) -> pd.DataFrame:
    # browse_conditions uses mesh_term, not name (name lives in conditions table)
    cols = "nct_id, mesh_term, downcase_mesh_term"
    return _chunked_query("browse_conditions", cols, nct_ids)


def get_interventions(nct_ids: list[str]) -> pd.DataFrame:
    cols = "nct_id, intervention_type, name"
    return _chunked_query("interventions", cols, nct_ids)


def get_calculated_values(nct_ids: list[str]) -> pd.DataFrame:
    # Only number_of_facilities — never actual_duration or were_results_reported (blocklisted)
    cols = "nct_id, number_of_facilities"
    return _chunked_query("calculated_values", cols, nct_ids)


def get_countries(nct_ids: list[str]) -> pd.DataFrame:
    # Registration-time trial geography. Pull name only; the `removed` flag marks a
    # post-registration removal, so num_countries counts all declared countries (ignores it).
    cols = "nct_id, name"
    return _chunked_query("countries", cols, nct_ids)


def test_connection() -> dict[str, Any]:
    """Return count of studies updated in the last 7 days; confirm SSL."""
    cfg = _load_aact_config()
    schema = cfg["schema"]
    q = text(
        f"SELECT COUNT(*) AS cnt FROM {schema}.studies "  # noqa: S608
        f"WHERE last_update_posted_date > NOW() - INTERVAL '7 days'"
    )
    try:
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(q).fetchone()
            count = row[0] if row else 0
        result = {"status": "ok", "studies_updated_last_7d": count, "ssl": cfg["sslmode"]}
        print(f"AACT connection OK. Studies updated in last 7 days: {count}")
        return result
    except Exception as exc:
        _handle_connection_error(exc)
        raise


def _handle_connection_error(exc: Exception) -> None:
    import psycopg2

    if isinstance(exc.__cause__, psycopg2.OperationalError) or "password" in str(exc).lower():
        logger.error(
            "AACT connection failed: %s\n"
            "→ Check AACT_USER / AACT_PASSWORD in your .env file.\n"
            "→ Register at %s if you don't have an account.",
            exc,
            _AACT_SIGNUP_URL,
        )
    else:
        logger.error("AACT query failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test AACT connection")
    args = parser.parse_args()
    if args.test:
        test_connection()
