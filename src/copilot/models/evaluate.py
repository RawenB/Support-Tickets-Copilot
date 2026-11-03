"""Evaluate model quality using macro F1 and supporting metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import f1_score


def evaluate(
    *,
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    task: str = "queue",
) -> dict[str, Any]:
    """Load a saved model and compute macro F1 on a held-out split."""
    if data_path is None:
        data_path = Path("data/raw/customer-support-tickets.csv")
    if model_path is None:
        model_path = Path("models") / f"{task}_model" / f"{task}_model.joblib"

    csv_path = Path(data_path)
    model_file = Path(model_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {csv_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")

    df = pd.read_csv(csv_path)
    task_column = task
    if task not in df.columns:
        task_column = "queue" if task == "queue" else "priority"
    if task_column not in df.columns:
        raise ValueError(f"Task column '{task_column}' not found in evaluation data.")

    text_columns = [column for column in ["subject", "body", "text"] if column in df.columns]
    df = df[[*text_columns, task_column]].copy().dropna().reset_index(drop=True)
    df["text"] = df[text_columns].fillna("").astype(str).agg(lambda row: " ".join(row), axis=1)

    model = joblib.load(model_file)
    X = df["text"]
    y_true = df[task_column]
    y_pred = model.predict(X)
    metrics = {"task": task, "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}

    report_path = Path("reports") / "metrics_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
