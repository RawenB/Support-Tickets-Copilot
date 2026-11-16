"""Data contract for the DVC splits. The ``validate`` stage fails the pipeline on violations,
so a bad data version can never reach training.

Writes ``reports/data_validation.json`` (tracked as a DVC metric, so ``dvc metrics diff``
shows how the data changed between two commits).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from copilot.data.text import frame_texts

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_validation.json"
REQUIRED = ("subject", "body", "queue", "priority", "language")
PRIORITIES = {"critical", "high", "medium", "low", "very_low"}


def _check(results: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    results.append({"check": name, "passed": bool(passed), "detail": detail})


def validate(splits_dir: Path = SPLITS_DIR) -> dict[str, Any]:
    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8")) or {}
    language = str(params.get("language", "en"))
    min_queue = int(params.get("min_queue_count", 5))
    frames = {n: pd.read_csv(splits_dir / f"{n}.csv", keep_default_na=False) for n in ("train", "val", "test")}
    results: list[dict[str, Any]] = []
    total = sum(len(f) for f in frames.values())

    for name, frame in frames.items():
        missing = [c for c in REQUIRED if c not in frame.columns]
        _check(results, f"{name}: required columns", not missing, missing)
        if missing:
            continue
        _check(results, f"{name}: body never blank", frame["body"].astype(str).str.strip().ne("").all())
        _check(results, f"{name}: labels never blank",
               (frame["queue"].astype(str).str.strip().ne("") & frame["priority"].astype(str).str.strip().ne("")).all())
        bad_priority = sorted(set(frame["priority"]) - PRIORITIES)
        _check(results, f"{name}: priority on documented scale", not bad_priority, bad_priority)
        _check(results, f"{name}: single language", set(frame["language"].str.lower()) == {language})
        texts = pd.Series(frame_texts(frame))
        _check(results, f"{name}: no exact duplicate tickets", not texts.duplicated().any(),
               int(texts.duplicated().sum()))

    keys = {n: set(frame_texts(f)) for n, f in frames.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = len(keys[a] & keys[b])
        _check(results, f"no exact leakage {a}/{b}", overlap == 0, overlap)

    counts = frames["train"]["queue"].value_counts()
    rare = counts.drop(labels=["Other"], errors="ignore")
    _check(results, f"every train queue has >= {min_queue} rows", bool((rare >= min_queue).all()),
           {k: int(v) for k, v in rare[rare < min_queue].items()})
    for name in ("val", "test"):
        unseen = sorted(set(frames[name]["queue"]) - set(counts.index))
        _check(results, f"{name}: no queue unseen in train", not unseen, unseen)
    for name, ratio in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        actual = len(frames[name]) / max(total, 1)
        _check(results, f"{name}: size ratio ~{ratio}", abs(actual - ratio) < 0.03, round(actual, 4))

    lengths = pd.Series(frame_texts(frames["train"])).str.split().str.len()
    report = {
        "passed": all(r["passed"] for r in results),
        "n_checks": len(results),
        "n_failed": sum(not r["passed"] for r in results),
        "rows": {n: int(len(f)) for n, f in frames.items()},
        "n_queues": int(frames["train"]["queue"].nunique()),
        "n_priorities": int(frames["train"]["priority"].nunique()),
        "blank_subject_share": round(float(frames["train"]["subject"].astype(str).str.strip().eq("").mean()), 4),
        "train_tokens_mean": round(float(lengths.mean()), 2),
        "train_tokens_p95": float(lengths.quantile(0.95)),
        "checks": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = validate()
    for result in report["checks"]:
        if not result["passed"]:
            print(f"FAILED: {result['check']} -> {result['detail']}", file=sys.stderr)
    print(f"data validation: {report['n_checks'] - report['n_failed']}/{report['n_checks']} checks passed")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
