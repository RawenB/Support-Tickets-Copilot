"""Train queue and priority classification models for support triage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from copilot.utils.config import load_yaml
from copilot.utils.mlflow_utils import setup_mlflow
from copilot.utils.seed import set_seed


def train(
    *,
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    task: str = "queue",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a simple text classifier for queue or priority classification."""
    set_seed(42)

    if data_path is None:
        data_path = Path("data/raw/customer-support-tickets.csv")
    if output_dir is None:
        output_dir = Path("models") / f"{task}_model"

    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    config = (
        load_yaml(config_path or "configs/train_queue.yaml")
        if task == "queue"
        else load_yaml(config_path or "configs/train_priority.yaml")
    )
    config.get("train", {})
    metrics_path = Path("reports") / f"{task}_metrics.json"
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if task not in df.columns:
        task_label = "queue" if task == "queue" else "priority"
        if task_label not in df.columns:
            raise ValueError(f"Task column '{task_label}' not found in training data.")
        task_col = task_label
    else:
        task_col = task

    text_columns = [column for column in ["subject", "body", "text"] if column in df.columns]
    if not text_columns:
        raise ValueError(
            "Training data requires a subject/body/text column for text classification."
        )

    df = df[[*text_columns, task_col]].copy()
    df["text"] = (
        df[text_columns].fillna("").astype(str).agg(lambda series: " ".join(series), axis=1)
    )
    df = df[["text", task_col]].dropna().reset_index(drop=True)

    X = df["text"]
    y = df[task_col]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True, ngram_range=(1, 2), min_df=1, strip_accents="unicode"
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    model_path = model_dir / f"{task}_model.joblib"
    joblib.dump(pipeline, model_path)
    metrics = {
        "task": task,
        "macro_f1": float(val_f1),
        "val_size": int(len(X_val)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    setup_mlflow("support-copilot")
    with mlflow.start_run(run_name=f"{task}-classifier") as run:
        mlflow.log_param("task", task)
        mlflow.log_param("model_type", "tfidf_logistic_regression")
        mlflow.log_param("output_dir", str(model_dir))
        mlflow.log_metric("macro_f1", float(val_f1))
        mlflow.log_artifact(str(model_path), artifact_path="model")

    return {
        "task": task,
        "model_path": str(model_path),
        "macro_f1": float(val_f1),
        "mlflow_run_id": run.info.run_id,
    }
