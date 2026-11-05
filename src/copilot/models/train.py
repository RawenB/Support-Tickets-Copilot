"""Train, compare, track, and register real queue/priority classifiers."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from copilot.models.distilbert import train_distilbert
from copilot.utils.config import load_yaml
from copilot.utils.mlflow_utils import setup_mlflow
from copilot.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("support_copilot.models.train")


def _prepare_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    frame = frame.copy()
    if task not in frame.columns:
        raise ValueError(f"Task column '{task}' not found in training data.")
    text_columns = [column for column in ("subject", "body", "text") if column in frame.columns]
    if not text_columns:
        raise ValueError("Training data requires subject/body/text columns.")
    frame["text"] = (
        frame[text_columns].fillna("").astype(str).agg(
            lambda row: " ".join(str(part) for part in row if str(part).strip()), axis=1
        )
    )
    frame[task] = frame[task].fillna("").astype(str).str.strip().str.lower()
    frame = frame[frame[task].ne("")].copy()
    return frame[["text", task]].reset_index(drop=True)


def _split_small_frame(frame: pd.DataFrame, task: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        train, temp = train_test_split(
            frame, test_size=0.30, random_state=seed, stratify=frame[task]
        )
        val, test = train_test_split(
            temp, test_size=0.50, random_state=seed, stratify=temp[task]
        )
    except ValueError:
        train, temp = train_test_split(frame, test_size=0.30, random_state=seed)
        val, test = train_test_split(temp, test_size=0.50, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _load_frames(
    data_path: str | Path | None, task: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    if data_path is not None:
        frame = _prepare_frame(pd.read_csv(Path(data_path)), task)
        train, val, test = _split_small_frame(frame, task, seed)
        return train, val, test, False
    train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_path = PROJECT_ROOT / "data" / "splits" / "val.csv"
    test_path = PROJECT_ROOT / "data" / "splits" / "test.csv"
    missing = [path for path in (train_path, val_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Real splits missing: {missing}. Run `dvc repro` first.")
    return (
        _prepare_frame(pd.read_csv(train_path), task),
        _prepare_frame(pd.read_csv(val_path), task),
        _prepare_frame(pd.read_csv(test_path), task),
        True,
    )


def _score(model: Any, frame: pd.DataFrame, task: str) -> dict[str, Any]:
    predictions = model.predict(frame["text"].tolist())
    return {
        "macro_f1": float(f1_score(frame[task], predictions, average="macro")),
        "accuracy": float(accuracy_score(frame[task], predictions)),
        "n": int(len(frame)),
    }


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        min_df=1,
        max_features=500000,
        sublinear_tf=True,
        stop_words="english",
        strip_accents="unicode",
    )


def _candidate_pipelines(task: str) -> dict[str, Pipeline]:
    svc = CalibratedClassifierCV(
        LinearSVC(C=32, class_weight="balanced", max_iter=12000, loss="hinge"),
        cv=3,
        method="isotonic",
    )
    sgd = SGDClassifier(
        loss="log_loss",
        alpha=1e-6,
        max_iter=5000,
        random_state=42,
        class_weight="balanced",
    )
    nb = ComplementNB(alpha=0.1)
    ensemble_weights = (3, 1, 1) if task == "queue" else (1, 1, 1)
    return {
        "tfidf_linear_svc": Pipeline([("tfidf", _vectorizer()), ("clf", svc)]),
        "tfidf_sgd_log": Pipeline([("tfidf", _vectorizer()), ("clf", sgd)]),
        "tfidf_ensemble": Pipeline(
            [
                ("tfidf", _vectorizer()),
                (
                    "clf",
                    VotingClassifier(
                        [("svc", svc), ("sgd", sgd), ("nb", nb)],
                        voting="soft",
                        weights=ensemble_weights,
                    ),
                ),
            ]
        ),
    }


def _log_sklearn_candidate(
    *,
    task: str,
    model_type: str,
    pipeline: Pipeline,
    train_size: int,
    val_size: int,
    test_size: int,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    run_name: str | None = None,
) -> str | None:
    setup_mlflow(f"support-copilot-{task}")
    with mlflow.start_run(run_name=run_name or f"{task}-{model_type}") as run:
        mlflow.log_params(
            {
                "task": task,
                "model_type": model_type,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
            }
        )
        mlflow.log_metrics(
            {
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "n_test": test_metrics["n"],
            }
        )
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        return run.info.run_id


def _log_distilbert_candidate(
    *,
    task: str,
    result: dict[str, Any],
    train_size: int,
    test_size: int,
) -> str | None:
    setup_mlflow(f"support-copilot-{task}")
    with mlflow.start_run(run_name=f"{task}-distilbert") as run:
        mlflow.log_params(
            {
                "task": task,
                "model_type": "distilbert",
                "train_size": train_size,
                "test_size": test_size,
                "base_model": "distilbert-base-uncased",
            }
        )
        mlflow.log_metrics(
            {
                "val_macro_f1": result["val_macro_f1"],
                "val_accuracy": result["val_accuracy"],
                "test_macro_f1": result["macro_f1"],
                "test_accuracy": result["accuracy"],
                "n_test": result["n_test"],
            }
        )
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_dir = result["model_dir"]
            mlflow.transformers.log_model(
                transformers_model={
                    "model": AutoModelForSequenceClassification.from_pretrained(model_dir),
                    "tokenizer": AutoTokenizer.from_pretrained(model_dir),
                },
                artifact_path="model",
                task="text-classification",
            )
        except Exception:
            logger.exception("Could not log DistilBERT as an MLflow transformers model.")
            mlflow.log_artifact(result["model_dir"], artifact_path="model")
        return run.info.run_id


def _register_champion(task: str, run_id: str | None) -> str | None:
    if not run_id:
        return None
    client = setup_mlflow(f"support-copilot-{task}")
    name = f"support-copilot-{task}"
    try:
        client.create_registered_model(name)
    except Exception:
        pass
    try:
        version = client.create_model_version(
            name=name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        client.set_registered_model_alias(name, "champion", version.version)
        return str(version.version)
    except Exception:
        logger.exception("Could not register champion for %s", task)
        return None


def _write_metrics(
    *,
    task: str,
    champion: dict[str, Any],
    run_id: str | None,
    write_reports: bool,
    output_dir: Path,
) -> dict[str, Any]:
    payload = {
        "task": task,
        "split": "test",
        "model_type": champion["model_type"],
        "model_path": champion.get("model_path"),
        "macro_f1": champion["test_metrics"]["macro_f1"],
        "accuracy": champion["test_metrics"]["accuracy"],
        "n_test": champion["test_metrics"]["n"],
        "val_macro_f1": champion["val_metrics"]["macro_f1"],
        "val_accuracy": champion["val_metrics"]["accuracy"],
        "mlflow_run_id": run_id,
    }
    if write_reports:
        reports = PROJECT_ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / f"metrics_{task}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        (reports / f"{task}_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"metrics_{task}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return payload


def train(
    *,
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    task: str = "queue",
    config_path: str | Path | None = None,
    use_distilbert: bool | None = None,
) -> dict[str, Any]:
    """Train real candidates, select by validation F1, and register the champion."""
    seed = 42
    set_seed(seed)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "models" / f"{task}_model"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path or f"configs/train_{task}.yaml").get("train", {})
    train_frame, val_frame, test_frame, is_real = _load_frames(data_path, task, seed)
    if is_real and use_distilbert is None:
        use_distilbert = os.environ.get("COPILOT_SKIP_DISTILBERT", "").lower() not in {
            "1",
            "true",
            "yes",
        }
    use_distilbert = bool(use_distilbert) if use_distilbert is not None else False

    candidates: list[dict[str, Any]] = []
    for model_type, pipeline in _candidate_pipelines(task).items():
        pipeline.fit(train_frame["text"], train_frame[task])
        val_metrics = _score(pipeline, val_frame, task)
        test_metrics = _score(pipeline, test_frame, task)
        run_id = None
        if is_real:
            run_id = _log_sklearn_candidate(
                task=task,
                model_type=model_type,
                pipeline=pipeline,
                train_size=len(train_frame),
                val_size=len(val_frame),
                test_size=len(test_frame),
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )
        candidates.append(
            {
                "model_type": model_type,
                "model": pipeline,
                "model_path": None,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "run_id": run_id,
            }
        )

    if use_distilbert:
        try:
            distilbert_result = train_distilbert(
                train_frame,
                val_frame,
                test_frame,
                task=task,
                output_dir=output_dir,
                model_name=str(config.get("model_name", "distilbert-base-uncased")),
                epochs=int(config.get("epochs", 2)),
                batch_size=int(config.get("batch_size", 16)),
                learning_rate=float(config.get("learning_rate", 2e-5)),
                class_weights=bool(config.get("class_weights", False)),
                max_length=int(config.get("max_length", 128)),
                seed=seed,
            )
            run_id = _log_distilbert_candidate(
                task=task,
                result=distilbert_result,
                train_size=len(train_frame),
                test_size=len(test_frame),
            )
            candidates.append(
                {
                    "model_type": "distilbert",
                    "model": joblib.load(distilbert_result["model_path"]),
                    "model_path": distilbert_result["model_path"],
                    "val_metrics": {
                        "macro_f1": distilbert_result["val_macro_f1"],
                        "accuracy": distilbert_result["val_accuracy"],
                        "n": distilbert_result["val_size"],
                    },
                    "test_metrics": {
                        "macro_f1": distilbert_result["macro_f1"],
                        "accuracy": distilbert_result["accuracy"],
                        "n": distilbert_result["n_test"],
                    },
                    "run_id": run_id,
                }
            )
        except Exception:
            logger.exception("DistilBERT training failed; keeping classical candidates.")

    champion = max(candidates, key=lambda item: item["val_metrics"]["macro_f1"])
    if champion["model_type"] != "distilbert":
        final_pipeline = clone(champion["model"]).fit(
            pd.concat([train_frame, val_frame], ignore_index=True)["text"],
            pd.concat([train_frame, val_frame], ignore_index=True)[task],
        )
        final_test_metrics = _score(final_pipeline, test_frame, task)
        champion["model"] = final_pipeline
        champion["test_metrics"] = final_test_metrics
        if is_real:
            champion["run_id"] = _log_sklearn_candidate(
                task=task,
                model_type=champion["model_type"],
                pipeline=final_pipeline,
                train_size=len(train_frame) + len(val_frame),
                val_size=len(val_frame),
                test_size=len(test_frame),
                val_metrics=champion["val_metrics"],
                test_metrics=final_test_metrics,
                run_name=f"{task}-{champion['model_type']}-final",
            )
    champion_path = output_dir / f"{task}_model.joblib"
    joblib.dump(champion["model"], champion_path)
    champion["model_path"] = str(champion_path)
    (output_dir / "champion.json").write_text(
        json.dumps(
            {
                "task": task,
                "model_type": champion["model_type"],
                "model_path": str(champion_path),
                "mlflow_run_id": champion["run_id"],
                "macro_f1": champion["test_metrics"]["macro_f1"],
                "accuracy": champion["test_metrics"]["accuracy"],
                "n_test": champion["test_metrics"]["n"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    registry_version = _register_champion(task, champion["run_id"]) if is_real else None
    metrics = _write_metrics(
        task=task,
        champion=champion,
        run_id=champion["run_id"],
        write_reports=is_real,
        output_dir=output_dir,
    )
    return {
        "task": task,
        "model_path": str(champion_path),
        "model_type": champion["model_type"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "mlflow_run_id": champion["run_id"],
        "registry_version": registry_version,
        "candidates": [
            {
                "model_type": item["model_type"],
                "val_macro_f1": item["val_metrics"]["macro_f1"],
                "test_macro_f1": item["test_metrics"]["macro_f1"],
                "mlflow_run_id": item["run_id"],
            }
            for item in candidates
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["queue", "priority"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-distilbert", action="store_true")
    args = parser.parse_args()
    result = train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        task=args.task,
        use_distilbert=False if args.skip_distilbert else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
