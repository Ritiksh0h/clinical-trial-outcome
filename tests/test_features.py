from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path("data/processed")


@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_feature_files_exist():
    for phase in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            assert (PROCESSED / f"features_phase{phase}_{split}.parquet").exists()
            assert (PROCESSED / f"labels_phase{phase}_{split}.parquet").exists()


@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_no_leakage_in_processed_features():
    from cto.features.leakage import LEAKAGE_BLOCKLIST

    for phase in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            df = pd.read_parquet(PROCESSED / f"features_phase{phase}_{split}.parquet")
            leaked = set(df.columns) & LEAKAGE_BLOCKLIST
            assert not leaked, f"Phase {phase} {split}: leaked columns: {leaked}"


@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_y_not_in_feature_files():
    for phase in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            df = pd.read_parquet(PROCESSED / f"features_phase{phase}_{split}.parquet")
            assert (
                "y" not in df.columns
            ), f"Phase {phase} {split}: 'y' found in feature file — must be separate"


@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_train_larger_than_val_and_test_nonempty():
    for phase in [1, 2, 3]:
        n_train = len(pd.read_parquet(PROCESSED / f"features_phase{phase}_train.parquet"))
        n_val = len(pd.read_parquet(PROCESSED / f"features_phase{phase}_val.parquet"))
        n_test = len(pd.read_parquet(PROCESSED / f"features_phase{phase}_test.parquet"))
        assert n_train > n_val, f"Phase {phase}: train ({n_train}) not larger than val ({n_val})"
        assert n_val > 0, f"Phase {phase}: val split is empty"
        assert n_test > 0, f"Phase {phase}: test split is empty"


@pytest.mark.skipif(not PROCESSED.exists(), reason="Run dvc repro featurize first")
def test_feature_schema_consistent_across_splits():
    for phase in [1, 2, 3]:
        cols_train = set(
            pd.read_parquet(PROCESSED / f"features_phase{phase}_train.parquet").columns
        )
        cols_val = set(pd.read_parquet(PROCESSED / f"features_phase{phase}_val.parquet").columns)
        cols_test = set(pd.read_parquet(PROCESSED / f"features_phase{phase}_test.parquet").columns)
        assert (
            cols_train == cols_val == cols_test
        ), f"Phase {phase}: feature schemas differ across splits"
