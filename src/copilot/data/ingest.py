"""Ingest raw support-ticket data into the project data pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ingest(data: pd.DataFrame | str | Path | None = None) -> pd.DataFrame:
    """Return a DataFrame from in-memory data or a CSV path."""
    if data is None:
        raise ValueError("A data source or DataFrame is required for ingestion.")

    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    if path.exists():
        return pd.read_csv(path)

    raise TypeError("Unsupported ingestion input; expected DataFrame or CSV path.")


if __name__ == "__main__":
    ingest()
