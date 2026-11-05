"""Deterministic, CSV-only data pipeline for the support-ticket dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer-support-tickets.csv"
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "ingested.parquet"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"
DEDUPED_PATH = PROJECT_ROOT / "data" / "processed" / "deduped.parquet"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
AUDIT_PATH = REPORTS_DIR / "data_audit.json"
REQUIRED_COLUMNS = ("subject", "body", "queue", "priority", "language")
PRIORITY_MAP = {
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "very low": "very_low",
    "very_low": "very_low",
}


def _load_params() -> dict[str, Any]:
    path = PROJECT_ROOT / "params.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _clean_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text).strip()


def _normalize_priority(value: Any) -> str:
    key = _clean_text(value).replace("-", " ")
    return PRIORITY_MAP.get(key, key.replace(" ", "_"))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline dependency missing: {path}")
    return pd.read_parquet(path)


def ingest() -> pd.DataFrame:
    """Read the raw CSV and persist an untouched intermediate copy."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")
    frame = pd.read_csv(RAW_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    _write_parquet(frame, INTERIM_PATH)
    return frame


def clean() -> pd.DataFrame:
    """Normalize text/labels, keep English rows, and drop unusable records."""
    params = _load_params()
    language = str(params.get("language", "en")).lower()
    frame = _read_parquet(INTERIM_PATH).copy()
    frame["language"] = frame["language"].fillna("").astype(str).str.strip().str.lower()
    frame = frame[frame["language"].eq(language)].copy()
    for column in ("subject", "body"):
        frame[column] = frame[column].map(_clean_text)
    frame["queue"] = frame["queue"].map(_slug)
    frame["priority"] = frame["priority"].map(_normalize_priority)
    frame["text"] = (
        frame["subject"].fillna("").astype(str).str.strip()
        + " "
        + frame["body"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    frame = frame[
        frame["subject"].ne("")
        & frame["body"].ne("")
        & frame["queue"].ne("")
        & frame["priority"].ne("")
    ].copy()
    _write_parquet(frame, CLEANED_PATH)
    return frame


def dedupe() -> pd.DataFrame:
    """Remove exact duplicate ticket text before splitting."""
    frame = _read_parquet(CLEANED_PATH).copy()
    frame = frame.drop_duplicates(subset=["subject", "body"], keep="first")
    frame = frame.reset_index(drop=True)
    _write_parquet(frame, DEDUPED_PATH)
    return frame


def _merge_rare_queues(frame: pd.DataFrame, min_queue_count: int, train_ratio: float) -> pd.DataFrame:
    frame = frame.copy()
    counts = frame["queue"].value_counts()
    threshold = max(min_queue_count, int(math.ceil(min_queue_count / max(train_ratio, 0.01))))
    rare = counts[counts < threshold].index
    frame.loc[frame["queue"].isin(rare), "queue"] = "Other"
    return frame


def _split_groups(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split whole text groups so duplicate sequences cannot leak across splits."""
    work = frame.copy()
    work["_group"] = work["subject"].astype(str) + "||" + work["body"].astype(str)
    groups = (
        work.groupby("_group", sort=False)["queue"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else "Other")
        .rename("group_queue")
    )
    if len(groups) < 3:
        return (
            work.iloc[: max(1, int(round(len(work) * train_ratio)))].copy(),
            work.iloc[:0].copy(),
            work.iloc[:0].copy(),
        )

    temp_size = val_ratio + test_ratio
    try:
        train_groups, temp_groups = train_test_split(
            groups.index,
            test_size=temp_size,
            random_state=seed,
            stratify=groups.values,
        )
    except ValueError:
        train_groups, temp_groups = train_test_split(
            groups.index,
            test_size=temp_size,
            random_state=seed,
        )

    temp_group_frame = groups.loc[temp_groups]
    try:
        val_groups, test_groups = train_test_split(
            temp_group_frame.index,
            test_size=test_ratio / temp_size,
            random_state=seed,
            stratify=temp_group_frame.values,
        )
    except ValueError:
        val_groups, test_groups = train_test_split(
            temp_group_frame.index,
            test_size=test_ratio / temp_size,
            random_state=seed,
        )

    train = work[work["_group"].isin(train_groups)].drop(columns=["_group"]).copy()
    val = work[work["_group"].isin(val_groups)].drop(columns=["_group"]).copy()
    test = work[work["_group"].isin(test_groups)].drop(columns=["_group"]).copy()
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _enforce_queue_contract(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    min_queue_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge any queue that is too small in train, then keep val/test inside train queues."""
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train_counts = train["queue"].value_counts()
    too_small = set(train_counts[train_counts < min_queue_count].index) - {"Other"}
    if too_small:
        for frame in (train, val, test):
            frame.loc[frame["queue"].isin(too_small), "queue"] = "Other"
    train_queues = set(train["queue"].unique())
    for frame in (val, test):
        frame.loc[~frame["queue"].isin(train_queues), "queue"] = "Other"
    return train, val, test


def split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the real 70/15/15 train/val/test CSV splits and audit report."""
    params = _load_params()
    seed = int(params.get("seed", 42))
    split_params = params.get("split", {})
    train_ratio = float(split_params.get("train", 0.70))
    val_ratio = float(split_params.get("val", 0.15))
    test_ratio = float(split_params.get("test", 0.15))
    min_queue_count = int(params.get("min_queue_count", 5))
    frame = _read_parquet(DEDUPED_PATH)
    raw = _read_parquet(INTERIM_PATH)
    cleaned = _read_parquet(CLEANED_PATH)

    frame = _merge_rare_queues(frame, min_queue_count, train_ratio)
    train, val, test = _split_groups(
        frame,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train, val, test = _enforce_queue_contract(train, val, test, min_queue_count)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit = {
        "source": str(RAW_PATH.relative_to(PROJECT_ROOT)),
        "raw_rows": int(len(raw)),
        "english_rows": int(len(cleaned)),
        "duplicate_rows": int(len(cleaned) - len(frame)),
        "null_rows": int(
            raw[list(REQUIRED_COLUMNS)].isna().any(axis=1).sum()
        ),
        "queue_counts": {str(k): int(v) for k, v in train["queue"].value_counts().items()},
        "priority_counts": {
            str(k): int(v) for k, v in train["priority"].value_counts().items()
        },
        "cleaning_steps": [
            "read raw CSV",
            "filter language to English",
            "normalize subject/body whitespace",
            "normalize queue names",
            "normalize priority labels",
            "drop null or blank required fields",
            "deduplicate exact subject/body pairs",
            f"merge queues with fewer than {min_queue_count} train rows into Other",
            "stratified group-aware 70/15/15 split",
        ],
        "splits": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return train, val, test


def run(stage: str) -> None:
    actions = {
        "ingest": ingest,
        "clean": clean,
        "dedupe": dedupe,
        "split": split,
        "all": lambda: (ingest(), clean(), dedupe(), split()),
    }
    if stage not in actions:
        raise SystemExit(f"Unknown stage: {stage}")
    actions[stage]()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["ingest", "clean", "dedupe", "split", "all"])
    args = parser.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()
