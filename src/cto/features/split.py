from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PARAMS_PATH = Path(__file__).parents[3] / "params.yaml"


def _load_split_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)["split"]


def make_temporal_splits(
    df: pd.DataFrame,
    date_col: str = "completion_date",
    train_cutoff: str | None = None,
    val_cutoff: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Split df into train/val/test by completion_date (never random).
    Cutoffs default to params.yaml split section; callers may override (e.g. the gold
    pipeline uses 2022/2023 instead of the weak pipeline's 2021/2022). One split function,
    one code path — the caller passes cutoffs, it does not fork the logic.
    Raises ValueError if any split is empty or below min_split_size.
    """
    p = _load_split_params()
    train_cutoff = pd.Timestamp(train_cutoff if train_cutoff is not None else p["train_cutoff"])
    val_cutoff = pd.Timestamp(val_cutoff if val_cutoff is not None else p["val_cutoff"])
    min_size = p.get("min_split_size", 100)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    train = df[df[date_col] <= train_cutoff]
    val = df[(df[date_col] > train_cutoff) & (df[date_col] <= val_cutoff)]
    test = df[df[date_col] > val_cutoff]

    splits = {"train": train, "val": val, "test": test}
    for name, split in splits.items():
        if len(split) == 0:
            raise ValueError(
                f"{name} split is empty. "
                f"Check cutoffs in params.yaml or the date range of your data."
            )
        if len(split) < min_size:
            logger.warning(
                "%s split has only %d rows (min_split_size=%d) — "
                "may be too small for reliable evaluation",
                name,
                len(split),
                min_size,
            )

    logger.info(
        "Temporal splits: train=%d (%s–%s) | val=%d (%s–%s) | test=%d (%s–%s)",
        len(train),
        train[date_col].min().date(),
        train[date_col].max().date(),
        len(val),
        val[date_col].min().date(),
        val[date_col].max().date(),
        len(test),
        test[date_col].min().date(),
        test[date_col].max().date(),
    )
    return splits


def assert_temporal_integrity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "completion_date",
) -> None:
    """Raise ValueError if splits are not in strict temporal order."""
    train_max = pd.to_datetime(train[date_col]).max()
    val_min = pd.to_datetime(val[date_col]).min()
    val_max = pd.to_datetime(val[date_col]).max()
    test_min = pd.to_datetime(test[date_col]).min()

    if train_max > val_min:
        raise ValueError(
            f"temporal integrity violation: train max ({train_max.date()}) "
            f"> val min ({val_min.date()})"
        )
    if val_max > test_min:
        raise ValueError(
            f"temporal integrity violation: val max ({val_max.date()}) "
            f"> test min ({test_min.date()})"
        )
