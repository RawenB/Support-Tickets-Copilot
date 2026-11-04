"""Baseline model utilities and experiments for support classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def _prepare_training_frame(data: pd.DataFrame, task: str) -> pd.DataFrame:
    """Standardize raw ticket rows into text-plus-label training data."""
    frame = data.copy()
    task_column = task if task in frame.columns else ("queue" if task == "priority" else "priority")
    if task_column not in frame.columns:
        raise ValueError(f"Task column '{task}' not found in the data frame.")

    text_columns = [column for column in ("subject", "body", "text") if column in frame.columns]
    if not text_columns:
        raise ValueError("Training data needs a subject/body/text column for classification.")

    if "language" in frame.columns:
        frame = frame[frame["language"].fillna("").astype(str).str.lower().eq("en")].copy()

    frame = frame[[*text_columns, task_column]].dropna(subset=[task_column]).copy()
    frame["text"] = frame[text_columns].fillna("").astype(str).agg(
        lambda row: " ".join(part for part in row if part), axis=1
    )
    return frame[["text", task_column]].reset_index(drop=True)


def baseline(
    data: pd.DataFrame | str | Path,
    *,
    task: str = "queue",
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a lightweight TF-IDF + logistic-regression baseline for ticket triage."""
    if isinstance(data, (str, Path)):
        frame = pd.read_csv(Path(data))
    else:
        frame = data.copy()

    prepared = _prepare_training_frame(frame, task)
    X = prepared["text"]
    y = prepared[task]

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, _, y_val, _ = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    except ValueError:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    macro_f1 = float(f1_score(y_val, val_pred, average="macro"))

    target_path = (
        Path(model_path)
        if model_path
        else Path("models") / f"{task}_baseline_model" / f"{task}_baseline_model.joblib"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target_path)

    payload = {
        "task": task,
        "model_path": str(target_path),
        "macro_f1": macro_f1,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
    }
    report_path = Path("reports") / f"{task}_baseline_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
