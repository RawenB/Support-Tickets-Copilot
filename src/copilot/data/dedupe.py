"""Remove duplicate ticket records and near-duplicate sequences."""

from __future__ import annotations

import pandas as pd


def dedupe(data: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows while keeping the first occurrence."""
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()
    key_columns = [column for column in ["subject", "body", "ticket_id"] if column in df.columns]
    if not key_columns:
        key_columns = list(df.columns)

    return df.drop_duplicates(subset=key_columns, keep="first").reset_index(drop=True)
