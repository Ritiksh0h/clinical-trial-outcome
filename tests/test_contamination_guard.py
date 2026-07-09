"""Tests for the Track-B contamination guard (audit Item 5)."""

import json

import pandas as pd
import pytest

from cto.features.contamination_guard import (
    filter_weak_excluding_gold_test,
    load_gold_test_nct_ids,
    save_gold_test_nct_ids,
)


def _weak(nct_ids):
    return pd.DataFrame({"nct_id": nct_ids, "y": [1] * len(nct_ids)})


def test_removes_contaminated_rows():
    weak = _weak(["A", "B", "C", "D"])
    safe = filter_weak_excluding_gold_test(weak, ["B", "D"])
    assert set(safe["nct_id"]) == {"A", "C"}


def test_keeps_clean_rows_untouched():
    weak = _weak(["A", "B", "C"])
    safe = filter_weak_excluding_gold_test(weak, ["Z"])  # none overlap
    assert set(safe["nct_id"]) == {"A", "B", "C"}
    assert len(safe) == 3


def test_empty_gold_test_no_removal():
    weak = _weak(["A", "B"])
    safe = filter_weak_excluding_gold_test(weak, [])
    assert len(safe) == 2


def test_full_overlap_removes_all():
    weak = _weak(["A", "B"])
    safe = filter_weak_excluding_gold_test(weak, ["A", "B"])
    assert len(safe) == 0


def test_high_contamination_scenario():
    """Audit scenario: most gold-test trials also appear in weak → those rows excluded."""
    weak = _weak([f"NCT{i}" for i in range(1000)])
    # 913 of them are also gold-test (the 91.3% figure)
    gold_test = [f"NCT{i}" for i in range(913)]
    safe = filter_weak_excluding_gold_test(weak, gold_test)
    assert len(safe) == 87
    assert not set(safe["nct_id"]) & set(gold_test)


def test_load_raises_if_missing(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="single source of truth"):
        load_gold_test_nct_ids(missing)


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "gold_test_nct_ids.json"
    save_gold_test_nct_ids(["B", "A", "A", "C"], p)  # dedup + sort
    assert json.loads(p.read_text()) == ["A", "B", "C"]
    assert load_gold_test_nct_ids(p) == ["A", "B", "C"]
