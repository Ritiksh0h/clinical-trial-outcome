"""Load CTO labels from Hugging Face and produce clean label DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PARAMS_PATH = Path(__file__).parents[3] / "params.yaml"
_RAW_DIR = Path(__file__).parents[3] / "data" / "raw"

# Columns to keep from phase files (label-side only — no LF votes as features)
_PHASE_KEEP = {"nct_id", "pred_proba"}
# Columns to keep from the gold set
_GOLD_KEEP = {"nct_id", "labels"}


def derive_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pred_proba → binary y, drop pred_proba.
    Input df must have columns: nct_id, pred_proba.
    Returns df with columns: nct_id, y.
    """
    out = df.copy()
    out["y"] = (out["pred_proba"] >= 0.5).astype(int)
    out = out.drop(columns=["pred_proba"])
    return out


def _load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_phase(hf_config: str) -> pd.DataFrame:
    """Load one CTO phase CSV from HuggingFace, keep nct_id + pred_proba only."""
    from datasets import load_dataset

    params = _load_params()
    repo = params["ingest"]["cto_hf_repo"]
    ds = load_dataset(repo, hf_config, split="test", trust_remote_code=False)
    df = ds.to_pandas()

    # Keep only the columns we need; drop all LF vote columns
    keep = [c for c in _PHASE_KEEP if c in df.columns]
    df = df[keep].copy()

    # Dedup on nct_id — keep last (matches PHASE0.md spec)
    before = len(df)
    df = df.drop_duplicates(subset="nct_id", keep="last").reset_index(drop=True)
    logger.info("  %s: %d rows → %d after dedup", hf_config, before, len(df))
    return df


def load_gold() -> pd.DataFrame:
    """Load CTO human_labels gold set."""
    from datasets import load_dataset

    params = _load_params()
    repo = params["ingest"]["cto_hf_repo"]
    ds = load_dataset(repo, "human_labels", split="test", trust_remote_code=False)
    df = ds.to_pandas()

    keep = [c for c in _GOLD_KEEP if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={"labels": "y"})

    before = len(df)
    df = df.drop_duplicates(subset="nct_id", keep="last").reset_index(drop=True)
    logger.info("  gold: %d rows → %d after dedup", before, len(df))
    return df


def load_all(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Load all four CTO label files. Saves parquet to data/raw/ by default.
    Returns {"phase1": df, "phase2": df, "phase3": df, "gold": df}.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    params = _load_params()
    phase_configs = {
        c["name"]: c["hf_config"]
        for c in params["ingest"]["cto_phase_configs"]
        if c["name"] != "gold"
    }

    result: dict[str, pd.DataFrame] = {}

    for name, hf_config in phase_configs.items():
        logger.info("Loading CTO %s (%s)…", name, hf_config)
        df = load_phase(hf_config)
        df = derive_binary_label(df)
        result[name] = df
        if save:
            path = _RAW_DIR / f"cto_{name}.parquet"
            df.to_parquet(path, index=False)
            logger.info("  saved %s (%d rows)", path.name, len(df))

    logger.info("Loading CTO gold (human_labels)…")
    df_gold = load_gold()
    result["gold"] = df_gold
    if save:
        path = _RAW_DIR / "cto_gold.parquet"
        df_gold.to_parquet(path, index=False)
        logger.info("  saved %s (%d rows)", path.name, len(df_gold))

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_all(save=True)
    for name, df in data.items():
        print(f"{name}: {len(df)} rows, cols={list(df.columns)}")
