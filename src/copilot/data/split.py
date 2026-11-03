"""Split cleaned ticket data into deterministic train, validation, and test sets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split(
    data: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a simple deterministic 3-way split for support ticket modeling."""
    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    length = len(data)
    train_count = int(round(length * train_ratio))
    val_count = int(round(length * val_ratio))
    test_count = length - train_count - val_count

    if test_count < 0:
        raise ValueError("The split ratios produce an invalid test set size.")

    train_set = data.iloc[:train_count].reset_index(drop=True)
    val_end = train_count + val_count
    val_set = data.iloc[train_count:val_end].reset_index(drop=True)
    test_set = data.iloc[val_end:].reset_index(drop=True)
    return train_set, val_set, test_set
