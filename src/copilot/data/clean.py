"""Clean and normalize raw ticket records before downstream modeling."""

from __future__ import annotations

import pandas as pd


def clean(data: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    """Filter to the target language and normalize text columns for support data."""
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()
    if "language" in df.columns:
        df["language"] = df["language"].fillna("").astype(str).str.lower()
        df = df[df["language"].eq(language.lower())].copy()

    for column in ["subject", "body"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
            df[column] = df[column].str.replace(r"\s+", " ", regex=True)
            df[column] = df[column].str.lower()

    if {"subject", "body"}.issubset(df.columns):
        df["text"] = (
            df["subject"].fillna("").astype(str).str.strip()
            + " "
            + df["body"].fillna("").astype(str).str.strip()
        )
        df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    return df.reset_index(drop=True)
