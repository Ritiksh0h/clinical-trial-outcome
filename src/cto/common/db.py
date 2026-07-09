"""Stub predictions/metrics DB. SQLite locally, Render Postgres in prod."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_engine(url: str = "sqlite:///cto_metrics.db") -> Engine:
    return create_engine(url)
