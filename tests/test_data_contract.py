"""Contract tests on the REAL cleaned splits (skipped until `dvc repro` has produced them).

Real facts: 61,765 raw rows, 28,261 English, 52 raw queues, 5 priority levels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = Path(os.environ.get("SPLITS_DIR", ROOT / "data" / "splits"))
PRIORITIES = {"critical", "high", "medium", "low", "very_low"}

pytestmark = pytest.mark.skipif(
    not all((SPLITS / f"{name}.csv").exists() for name in ("train", "val", "test")),
    reason="splits not built yet: run `dvc repro`",
)


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(SPLITS / f"{n}.csv") for n in ("train", "val", "test")}


@pytest.fixture(scope="module")
def min_queue_count() -> int:
    return int(yaml.safe_load((ROOT / "params.yaml").read_text())["min_queue_count"])


def test_real_scale_not_a_toy_dataset(frames):
    assert len(frames["train"]) >= 5000, "train split is far too small for the real data"
    assert len(frames["val"]) >= 500 and len(frames["test"]) >= 500


def test_split_ratios_are_about_70_15_15(frames):
    total = sum(len(f) for f in frames.values())
    assert abs(len(frames["train"]) / total - 0.70) < 0.03
    assert abs(len(frames["val"]) / total - 0.15) < 0.03
    assert abs(len(frames["test"]) / total - 0.15) < 0.03


def test_only_english_and_no_missing_values(frames):
    for name, f in frames.items():
        if "language" in f:
            assert set(f["language"].str.lower()) == {"en"}, name
        for col in ("subject", "body", "queue", "priority"):
            assert f[col].notna().all(), f"{name}.{col} has nulls"
            assert (f[col].astype(str).str.strip() != "").all(), f"{name}.{col} has blanks"


def test_priority_values_use_the_documented_scale(frames):
    for name, f in frames.items():
        assert set(f["priority"].unique()) <= PRIORITIES, name


def test_no_text_leaks_between_splits(frames):
    def keys(f):
        return set((f["subject"].str.strip() + "||" + f["body"].str.strip()))

    assert not keys(frames["train"]) & keys(frames["test"]), "train/test leakage"
    assert not keys(frames["train"]) & keys(frames["val"]), "train/val leakage"
    assert not keys(frames["val"]) & keys(frames["test"]), "val/test leakage"


def test_no_exact_duplicates_inside_a_split(frames):
    for name, f in frames.items():
        assert not f.duplicated(["subject", "body"]).any(), name


def test_rare_queues_are_merged_and_unseen_queues_do_not_exist(frames, min_queue_count):
    counts = frames["train"]["queue"].value_counts()
    real_queues = counts.drop(labels=["Other"], errors="ignore")
    assert (real_queues >= min_queue_count).all(), "rare queues were not merged into Other"
    train_queues = set(counts.index)
    assert set(frames["val"]["queue"]) <= train_queues
    assert set(frames["test"]["queue"]) <= train_queues


def test_split_is_stratified_for_the_main_queues(frames):
    train = frames["train"]["queue"].value_counts(normalize=True)
    test = frames["test"]["queue"].value_counts(normalize=True)
    for queue in train.head(8).index:
        assert abs(train[queue] - test.get(queue, 0)) < 0.03, f"{queue} not stratified"


def test_audit_report_documents_every_cleaning_step():
    audit = json.loads((ROOT / "reports" / "data_audit.json").read_text())
    text = json.dumps(audit).lower()
    for word in ("raw", "english", "duplicate", "null", "queue"):
        assert word in text, f"data_audit.json does not mention '{word}'"
