from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from copilot.utils.mlflow_utils import setup_mlflow

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "huggingface" / "customer-support-tickets.csv"
MODEL_PATH = ROOT / "models" / "baseline_priority_model.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
EXPERIMENT = "tickets-copilot-priority-baseline"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[["subject", "body", "priority", "language"]].copy()
    df = df.dropna(subset=["subject", "body", "priority", "language"])
    df["language"] = df["language"].astype(str)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = df["subject"].str.strip() + " " + df["body"].str.strip()
    return df[df["language"] == "en"].reset_index(drop=True)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def main() -> None:
    setup_mlflow(EXPERIMENT)
    df = load_data()
    X = df["text"]
    y = df["priority"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    model = build_model()

    with mlflow.start_run(run_name="priority-classifier-baseline") as run:
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        test_f1 = f1_score(y_test, test_pred, average="macro")

        mlflow.log_param("model_type", "tfidf_logistic_regression")
        mlflow.log_param("language_filter", "en")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("random_state", 42)
        mlflow.log_param("target_column", "priority")

        mlflow.log_metric("validation_accuracy", val_acc)
        mlflow.log_metric("validation_macro_f1", val_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_macro_f1", test_f1)

        print(f"Train size: {len(X_train)}")
        print(f"Validation size: {len(X_val)}")
        print(f"Test size: {len(X_test)}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Validation macro F1: {val_f1:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test macro F1: {test_f1:.4f}")
        print("\nTest classification report:\n")
        print(classification_report(y_test, test_pred))

        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")
        print(f"Model saved to: {MODEL_PATH}")
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
