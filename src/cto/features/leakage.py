"""
Leakage gate — the single most important module in this project.
Every feature-building function MUST call assert_no_leakage() before returning.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yaml

# src/cto/features/ -> src/cto/ -> src/ -> project_root
_BLOCKLIST_PATH = Path(__file__).parents[3] / "config" / "leakage_blocklist.yaml"


def _load_blocklist() -> set[str]:
    with open(_BLOCKLIST_PATH) as f:
        raw = yaml.safe_load(f)
    columns: set[str] = set()
    for section_cols in raw.values():
        if isinstance(section_cols, list):
            columns.update(section_cols)
    return columns


LEAKAGE_BLOCKLIST: set[str] = _load_blocklist()


def assert_no_leakage(df: pd.DataFrame, context: str = "") -> None:
    """
    Raise ValueError if any column in df is in LEAKAGE_BLOCKLIST.
    Call this at the end of every feature-building function.

    Raises:
        ValueError: If any post-hoc column survives into the feature matrix.
    """
    leaked = set(df.columns) & LEAKAGE_BLOCKLIST
    if leaked:
        raise ValueError(
            f"LEAKAGE DETECTED{' in ' + context if context else ''}. "
            f"The following post-hoc columns must never appear in features: {sorted(leaked)}\n"
            f"Check config/leakage_blocklist.yaml and remove these columns "
            f"from the feature-building pipeline."
        )


def drop_leakage_columns(df: pd.DataFrame, warn: bool = True) -> pd.DataFrame:
    """
    Drop all blocklisted columns from df and return the cleaned DataFrame.
    Prefer assert_no_leakage() (fail-fast); use this only for raw data cleaning
    before feature engineering.
    """
    to_drop = [c for c in df.columns if c in LEAKAGE_BLOCKLIST]
    if to_drop and warn:
        warnings.warn(
            f"Dropping {len(to_drop)} leakage column(s): {to_drop}. "
            f"These should never enter the feature pipeline.",
            stacklevel=2,
        )
    return df.drop(columns=to_drop)
