import numpy as np
import pandas as pd
import pytest

from cto.features.split import assert_temporal_integrity, make_temporal_splits


def make_fake_df(n=300):
    dates = pd.date_range("2015-01-01", "2024-12-31", periods=n)
    return pd.DataFrame(
        {
            "nct_id": [f"NCT{i:07d}" for i in range(n)],
            "completion_date": dates,
            "y": np.random.randint(0, 2, n),
        }
    )


def test_splits_are_nonempty():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    for name, split in splits.items():
        assert len(split) > 0, f"{name} split is empty"


def test_temporal_integrity_holds():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    assert_temporal_integrity(
        splits["train"], splits["val"], splits["test"], date_col="completion_date"
    )  # must not raise


def test_no_row_appears_in_two_splits():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    all_ids = [set(s["nct_id"]) for s in splits.values()]
    for i, s1 in enumerate(all_ids):
        for j, s2 in enumerate(all_ids):
            if i != j:
                overlap = s1 & s2
                assert not overlap, f"Splits {i} and {j} share nct_ids: {overlap}"


def test_train_predates_val():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    assert splits["train"]["completion_date"].max() <= splits["val"]["completion_date"].min()


def test_val_predates_test():
    df = make_fake_df(300)
    splits = make_temporal_splits(df)
    assert splits["val"]["completion_date"].max() <= splits["test"]["completion_date"].min()


def test_temporal_integrity_raises_on_violation():
    train = pd.DataFrame({"nct_id": ["A"], "completion_date": pd.to_datetime(["2023-01-01"])})
    val = pd.DataFrame(
        {"nct_id": ["B"], "completion_date": pd.to_datetime(["2020-01-01"])}
    )  # before train!
    test = pd.DataFrame({"nct_id": ["C"], "completion_date": pd.to_datetime(["2024-01-01"])})
    with pytest.raises(ValueError, match="temporal"):
        assert_temporal_integrity(train, val, test, "completion_date")


def test_empty_split_raises():
    df = pd.DataFrame(
        {
            "nct_id": ["A", "B"],
            "completion_date": pd.to_datetime(["2015-01-01", "2016-01-01"]),
            "y": [0, 1],
        }
    )
    with pytest.raises(ValueError):
        make_temporal_splits(df)  # val and test will be empty
