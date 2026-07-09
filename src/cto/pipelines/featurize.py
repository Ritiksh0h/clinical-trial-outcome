"""Featurize pipeline — produces per-phase train/val/test feature matrices."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"
_PARAMS_PATH = Path(__file__).parents[3] / "params.yaml"


def run() -> None:
    from cto.features.build import build_features, build_raw_joined
    from cto.features.split import assert_temporal_integrity, make_temporal_splits

    with open(_PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    phases = params["features"]["phases"]

    for phase in phases:
        logger.info("=== Phase %d ===", phase)
        df = build_raw_joined(phase)
        splits = make_temporal_splits(df)
        assert_temporal_integrity(splits["train"], splits["val"], splits["test"])

        for split_name in ["train", "val", "test"]:
            split_df = splits[split_name]
            y = split_df["y"].reset_index(drop=True)

            X = build_features(phase, split_name, df=split_df)

            feat_path = _PROCESSED_DIR / f"features_phase{phase}_{split_name}.parquet"
            label_path = _PROCESSED_DIR / f"labels_phase{phase}_{split_name}.parquet"
            X.to_parquet(feat_path, index=False)
            y.to_frame().to_parquet(label_path, index=False)

            logger.info(
                "  Phase %d %s: X=%s, y=%d (pos=%.2f)",
                phase,
                split_name,
                X.shape,
                len(y),
                y.mean(),
            )

    logger.info("Featurize complete. Files in %s", _PROCESSED_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
