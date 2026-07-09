"""TF-IDF features from eligibility criteria text."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

_TFIDF_PATH = Path(__file__).parents[3] / "models" / "tfidf_vectorizer.joblib"


def fit_tfidf(texts: pd.Series, max_features: int, ngram_range: tuple[int, int]) -> TfidfVectorizer:
    """Fit on training texts only. Saves to models/tfidf_vectorizer.joblib."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    vec.fit(texts.fillna("").tolist())
    _TFIDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, _TFIDF_PATH)
    logger.info("fit_tfidf: vocab=%d, saved to %s", len(vec.vocabulary_), _TFIDF_PATH)
    return vec


def transform_tfidf(texts: pd.Series, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    """Transform texts; returns DataFrame with columns tfidf_0…tfidf_{n-1}."""
    n = vectorizer.max_features or len(vectorizer.vocabulary_)
    matrix = vectorizer.transform(texts.fillna("").tolist())
    cols = [f"tfidf_{i}" for i in range(n)]
    return pd.DataFrame(matrix.toarray(), columns=cols, index=texts.index)


def load_tfidf() -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer from disk."""
    if not _TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {_TFIDF_PATH}. "
            "Run featurize for phase=1, split='train' first."
        )
    return joblib.load(_TFIDF_PATH)
