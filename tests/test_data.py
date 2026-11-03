from __future__ import annotations

import pandas as pd

from copilot.data.clean import clean
from copilot.data.dedupe import dedupe
from copilot.data.ingest import ingest
from copilot.data.split import split


def test_ingest_reads_csv_and_returns_dataframe() -> None:
    df = pd.DataFrame(
        {
            "ticket_id": [1, 2],
            "subject": ["Login issue", "Password reset"],
            "body": ["Cannot sign in", "Reset my password"],
            "priority": ["high", "low"],
            "queue": ["technical", "account"],
            "language": ["en", "en"],
        }
    )
    result = ingest(df)
    assert list(result.columns) == list(df.columns)
    assert len(result) == 2


def test_clean_filters_language_and_normalizes_text() -> None:
    raw = pd.DataFrame(
        {
            "subject": [" Login issue ", None, "PASSWORD reset"],
            "body": ["Cannot sign in", "Please help", "Need reset"],
            "priority": ["high", "medium", "low"],
            "queue": ["technical", "billing", "account"],
            "language": ["en", "fr", "en"],
        }
    )
    cleaned = clean(raw, language="en")
    assert len(cleaned) == 2
    assert cleaned["subject"].tolist() == ["login issue", "password reset"]
    assert cleaned["text"].str.contains("login").any()


def test_dedupe_removes_exact_duplicates() -> None:
    raw = pd.DataFrame(
        {
            "subject": ["login issue", "login issue", "password reset"],
            "body": ["cannot sign in", "cannot sign in", "need reset"],
            "priority": ["high", "high", "low"],
            "queue": ["technical", "technical", "account"],
            "language": ["en", "en", "en"],
        }
    )
    deduped = dedupe(raw)
    assert len(deduped) == 2
    assert deduped["subject"].nunique() == 2


def test_split_returns_expected_parts() -> None:
    df = pd.DataFrame({"text": [f"ticket {i}" for i in range(20)], "priority": ["high"] * 20})
    train, val, test = split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert len(train) + len(val) + len(test) == 20
    assert len(train) == 14
    assert len(val) + len(test) == 6
