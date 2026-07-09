"""Data loader tests — no network access required."""

import pandas as pd

from cto.data.cto_labels import derive_binary_label


def test_cto_labels_schema():
    """Minimal schema check — does not require network access."""
    df = pd.DataFrame({"nct_id": ["NCT001", "NCT002"], "pred_proba": [0.3, 0.8]})
    out = derive_binary_label(df)
    assert "y" in out.columns
    assert out["y"].isin([0, 1]).all()
    assert "pred_proba" not in out.columns  # dropped after deriving y


def test_derive_binary_label_threshold():
    """Exactly 0.5 rounds to success (>=)."""
    df = pd.DataFrame({"nct_id": ["A", "B", "C"], "pred_proba": [0.0, 0.5, 1.0]})
    out = derive_binary_label(df)
    assert list(out["y"]) == [0, 1, 1]


def test_derive_binary_label_preserves_nct_id():
    df = pd.DataFrame({"nct_id": ["NCT9999"], "pred_proba": [0.7]})
    out = derive_binary_label(df)
    assert out["nct_id"].iloc[0] == "NCT9999"
