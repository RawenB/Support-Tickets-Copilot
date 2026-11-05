"""Evaluate model quality using macro F1 and supporting metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _prepare_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if task not in frame.columns:
        raise ValueError(f"Task column '{task}' not found in evaluation data.")
    text_columns = [column for column in ("subject", "body", "text") if column in frame.columns]
    if not text_columns:
        raise ValueError("Evaluation data requires subject/body/text columns.")
    frame = frame.copy()
    frame["text"] = (
        frame[text_columns].fillna("").astype(str).agg(
            lambda row: " ".join(str(part) for part in row if str(part).strip()), axis=1
        )
    )
    frame[task] = frame[task].fillna("").astype(str).str.strip().str.lower()
    return frame[frame[task].ne("")].reset_index(drop=True)


def _model_type(task: str) -> str:
    champion = PROJECT_ROOT / "models" / f"{task}_model" / "champion.json"
    if champion.exists():
        try:
            return str(json.loads(champion.read_text(encoding="utf-8")).get("model_type", "unknown"))
        except Exception:
            pass
    return "unknown"


def evaluate(
    *,
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    task: str = "queue",
    write_report: bool | None = None,
) -> dict[str, Any]:
    """Load a saved model and compute macro F1 on the real held-out test split."""
    is_real_test = data_path is None
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "splits" / "test.csv"
        if write_report is None:
            write_report = True
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / f"{task}_model" / f"{task}_model.joblib"

    csv_path = Path(data_path)
    model_file = Path(model_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {csv_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")

    frame = _prepare_frame(pd.read_csv(csv_path), task)
    model = joblib.load(model_file)
    predictions = model.predict(frame["text"].tolist())
    existing: dict[str, Any] = {}
    metrics_path = PROJECT_ROOT / "reports" / f"metrics_{task}.json"
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    payload = {
        **existing,
        "task": task,
        "split": "test" if is_real_test or "test" in str(csv_path) else "custom",
        "model_type": _model_type(task),
        "macro_f1": float(f1_score(frame[task], predictions, average="macro")),
        "accuracy": float(accuracy_score(frame[task], predictions)),
        "n_test": int(len(frame)),
    }

    if write_report is None:
        write_report = False
    if write_report:
        reports = PROJECT_ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / f"metrics_{task}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["queue", "priority"], default=None)
    args = parser.parse_args()
    tasks = [args.task] if args.task else ["queue", "priority"]
    summary: dict[str, Any] = {}
    for task in tasks:
        summary[task] = evaluate(task=task)
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
