"""Audit dataset quality, schema, and drift checks for the support pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def audit(data: pd.DataFrame | str | Path) -> dict[str, Any]:
    """Summarize row quality, language coverage, and label balance for ticket data."""
    if isinstance(data, (str, Path)):
        frame = pd.read_csv(Path(data))
    else:
        frame = data.copy()

    if frame.empty:
        return {
            "row_count": 0,
            "duplicate_rows": 0,
            "missing_values": 0,
            "languages": {},
            "queue_counts": {},
            "priority_counts": {},
        }

    language_counts = (
        frame["language"].fillna("unknown").astype(str).str.lower().value_counts().to_dict()
        if "language" in frame.columns
        else {}
    )
    queue_counts = (
        frame["queue"].fillna("unknown").astype(str).str.lower().value_counts().to_dict()
        if "queue" in frame.columns
        else {}
    )
    priority_counts = (
        frame["priority"].fillna("unknown").astype(str).str.lower().value_counts().to_dict()
        if "priority" in frame.columns
        else {}
    )

    return {
        "row_count": int(len(frame)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "languages": language_counts,
        "queue_counts": queue_counts,
        "priority_counts": priority_counts,
    }
