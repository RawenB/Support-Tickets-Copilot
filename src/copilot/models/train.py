"""Train, compare, track and register the queue / priority classifiers.

Flow for one task (``python -m copilot.models.train --task queue``):

1. Load the DVC-versioned splits and build model inputs with the shared ``build_text``.
2. Fit every candidate in ``configs/train_<task>.yaml`` on *train* and score it on *val*.
   Each candidate is an MLflow child run (params, per-epoch curves, val metrics).
3. Pick the champion by validation macro F1. The test split is touched exactly once,
   for the champion only, so the reported test score is an unbiased estimate.
4. Write ``reports/metrics_<task>.json`` (+ per-class report, confusion plot data) and the
   model folder that DVC versions, then log the model to MLflow with full lineage tags
   and register it as the ``challenger``. ``copilot.registry.promote`` moves ``champion``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from copilot.data.text import frame_texts
from copilot.models.artifacts import relative_to_root, save_model
from copilot.models.blend import BlendedTicketModel, aligned_proba, tune_weights
from copilot.models.knn import KNNTicketModel
from copilot.models.metrics import (
    classification_metrics,
    confusion_figure,
    confusion_rows,
    novelty_scores,
    per_class_report,
)
from copilot.utils.config import load_yaml
from copilot.utils.lineage import lineage_tags
from copilot.utils.mlflow_utils import setup_mlflow
from copilot.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
logger = logging.getLogger("support_copilot.models.train")
DEFAULT_CANDIDATES = ("tfidf_sgd", "knn_char", "knn_word", "knn_embedding", "ensemble")


# --------------------------------------------------------------------------- data


def _labels_of(frame: pd.DataFrame, task: str) -> pd.Series:
    return frame[task].fillna("").astype(str).str.strip().str.lower()


def _prepare(frame: pd.DataFrame, task: str) -> tuple[list[str], list[str]]:
    if task not in frame.columns:
        raise ValueError(f"Task column '{task}' not found in training data.")
    labels = _labels_of(frame, task)
    keep = labels.ne("").to_numpy()
    texts = frame_texts(frame.loc[keep].reset_index(drop=True))
    return texts, labels[keep].tolist()


def _load_splits(data_path: str | Path | None, task: str, seed: int):
    """Real DVC splits, or a small stratified split of an ad-hoc CSV (used by tests)."""
    if data_path is None:
        paths = [SPLITS_DIR / f"{name}.csv" for name in ("train", "val", "test")]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Real splits missing: {missing}. Run `dvc repro split` first.")
        return [_prepare(pd.read_csv(p), task) for p in paths], True, paths
    frame = pd.read_csv(data_path)
    try:
        train, temp = train_test_split(frame, test_size=0.3, random_state=seed, stratify=frame[task])
        val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp[task])
    except ValueError:
        train, temp = train_test_split(frame, test_size=0.3, random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    return [_prepare(f.reset_index(drop=True), task) for f in (train, val, test)], False, []


# --------------------------------------------------------------------------- candidates


def _tfidf_pipeline(cfg: dict[str, Any], seed: int) -> Pipeline:
    """High-capacity word n-gram model: excellent at recognising paraphrased repeat tickets."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, int(cfg.get("ngram_max", 3))),
                    min_df=int(cfg.get("min_df", 1)),
                    max_features=int(cfg.get("max_features", 500000)),
                    sublinear_tf=True,
                    stop_words="english",
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=float(cfg.get("alpha", 1e-6)),
                    max_iter=5000,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _knn_member(name: str, cfg: dict[str, Any]) -> KNNTicketModel:
    representation = name.split("_", 1)[1]
    defaults = {"char": {"k": 15, "power": 10}, "word": {"k": 15, "power": 4},
                "embedding": {"k": 5, "power": 10}}[representation]
    options = {**defaults, **cfg.get(representation, {})}
    return KNNTicketModel(representation, **options)


def _log_dict(payload: Any, name: str) -> None:
    mlflow.log_dict(payload, name)


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(f"{prefix}.{key}" if prefix else str(key), item))
        return out
    if isinstance(value, (list, tuple)):
        return {prefix: ",".join(str(v) for v in value)}
    return {prefix: value}


# --------------------------------------------------------------------------- outputs


def _write_reports(task: str, payload: dict[str, Any], per_class: dict[str, Any], rows: list[dict[str, str]],
                   history: list[dict[str, float]], output_dir: Path, write_reports: bool) -> None:
    target = REPORTS_DIR if write_reports else output_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / f"metrics_{task}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not write_reports:
        return
    (target / f"per_class_{task}.json").write_text(json.dumps(per_class, indent=2), encoding="utf-8")
    plots = target / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / f"confusion_{task}.json").write_text(json.dumps(rows), encoding="utf-8")
    (plots / f"training_curve_{task}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def _log_model_to_mlflow(task: str, model_file: Path, output_dir: Path, sample: list[str]) -> None:
    """Log the champion as an MLflow pyfunc so it can be loaded or served from the registry."""
    from copilot.models.pyfunc import TicketClassifierPyfunc

    artifacts = {"model_file": str(model_file)}
    transformer_dir = output_dir / "transformer"
    if transformer_dir.exists():
        artifacts["transformer"] = str(transformer_dir)
    signature = mlflow.models.infer_signature(
        pd.DataFrame({"text": sample}),
        pd.DataFrame({"label": ["x"] * len(sample), "confidence": [0.5] * len(sample)}),
    )
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=TicketClassifierPyfunc(),
        artifacts=artifacts,
        code_paths=[str(PROJECT_ROOT / "src" / "copilot")],
        signature=signature,
        input_example=pd.DataFrame({"text": sample[:2]}),
        pip_requirements=["scikit-learn", "torch", "transformers", "pandas", "numpy", "joblib"],
    )


def _register(task: str, run_id: str, tags: dict[str, str], description: str) -> str | None:
    client = setup_mlflow(f"support-copilot-{task}")
    name = f"support-copilot-{task}"
    try:
        client.create_registered_model(name)
    except Exception:
        pass
    try:
        version = client.create_model_version(
            name=name, source=f"runs:/{run_id}/model", run_id=run_id, tags=tags, description=description
        )
        client.set_registered_model_alias(name, "challenger", version.version)
        return str(version.version)
    except Exception:
        logger.exception("Could not register challenger for %s", task)
        return None


# --------------------------------------------------------------------------- main entry


def train(
    *,
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    task: str = "queue",
    config_path: str | Path | None = None,
    use_transformer: bool | None = None,
    use_distilbert: bool | None = None,
) -> dict[str, Any]:
    """Train every configured candidate, pick a champion on validation F1, track and register it."""
    params = load_yaml("params.yaml")
    seed = int(params.get("seed", 42))
    set_seed(seed)
    config = load_yaml(config_path or f"configs/train_{task}.yaml").get("train", {})
    (train_xy, val_xy, test_xy), is_real, split_paths = _load_splits(data_path, task, seed)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train_xy, val_xy, test_xy
    labels = sorted(set(y_train))

    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / config.get("output_dir", f"models/{task}_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_transformer is None:
        use_transformer = use_distilbert
    if use_transformer is None:
        skip = {os.environ.get(k, "").lower() for k in ("COPILOT_SKIP_TRANSFORMER", "COPILOT_SKIP_DISTILBERT")}
        use_transformer = is_real and not ({"1", "true", "yes"} & skip)
    wanted = [c for c in config.get("candidates", DEFAULT_CANDIDATES) if c != "transformer" or use_transformer]
    if not is_real:  # toy CSVs (tests): keep it to fast, dependency-free members
        wanted = [c for c in wanted if c in {"tfidf_sgd", "knn_char", "knn_word", "ensemble"}]

    track = is_real
    tags = lineage_tags(split_paths) if is_real else {}
    parent = None
    if track:
        setup_mlflow(f"support-copilot-{task}")
        parent = mlflow.start_run(run_name=f"{task}-train")
        mlflow.set_tags({**tags, "task": task, "stage": "train"})
        mlflow.log_params(_flatten("", config))
        mlflow.log_params({"seed": seed, "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
                           "n_labels": len(labels)})
        _log_dict({k: int(v) for k, v in pd.Series(y_train).value_counts().items()}, "data/train_label_counts.json")

    candidates: dict[str, dict[str, Any]] = {}
    val_proba: dict[str, np.ndarray] = {}
    history: list[dict[str, float]] = []
    try:
        # -- 1. memory-style members, fitted on train only so validation stays unseen
        for name in wanted:
            if name == "tfidf_sgd":
                member = _tfidf_pipeline(config.get("tfidf", {}), seed)
            elif name.startswith("knn_"):
                member = _knn_member(name, config.get("knn", {}))
            else:
                continue
            member.fit(x_train, y_train)
            val_proba[name] = aligned_proba(member, x_val, labels)
            candidates[name] = {"model": member, "refittable": True}

        # -- 2. optional fine-tuned transformer (kept for experiments; see docs/MODEL_CARDS.md)
        if "transformer" in wanted:
            from copilot.models.transformer import fine_tune

            tcfg = dict(config.get("transformer", {}))
            child = mlflow.start_run(run_name=f"{task}-transformer", nested=True) if track else None
            if track:
                mlflow.log_params(_flatten("transformer", tcfg))

            def _on_epoch(epoch: int, record: dict[str, float]) -> None:
                if track:
                    mlflow.log_metrics({k: v for k, v in record.items() if k != "epoch"}, step=epoch)

            try:
                result = fine_tune(x_train, y_train, x_val, y_val, output_dir=output_dir, seed=seed,
                                   on_epoch_end=_on_epoch, **tcfg)
                history = result["history"]
                val_proba["transformer"] = aligned_proba(result["model"], x_val, labels)
                candidates["transformer"] = {"model": result["model"], "refittable": False}
                if track:
                    mlflow.log_metric("best_epoch", result["best_epoch"])
            finally:
                if child is not None:
                    mlflow.end_run()

        # -- 3. ensemble: weights over the members tuned on validation macro F1
        ecfg = config.get("ensemble", {})
        members = [m for m in ecfg.get("members", list(val_proba)) if m in val_proba]
        if "ensemble" in wanted and len(members) >= 2:
            weights, _ = tune_weights({m: val_proba[m] for m in members}, y_val, labels,
                                      step=float(ecfg.get("grid_step", 0.1)))
            weights = {m: w for m, w in weights.items() if w > 0}
            val_proba["ensemble"] = sum(w * val_proba[m] for m, w in weights.items())
            candidates["ensemble"] = {
                "model": BlendedTicketModel({m: candidates[m]["model"] for m in weights}, weights, labels),
                "weights": weights,
            }

        # -- validation scores for every candidate (selection data only)
        label_array = np.asarray(labels)
        for name, proba in val_proba.items():
            metrics = classification_metrics(y_val, label_array[proba.argmax(axis=1)], labels=labels, prefix="val_")
            candidates[name]["val"] = metrics
            if track:
                with mlflow.start_run(run_name=f"{task}-{name}", nested=True):
                    mlflow.set_tags({**tags, "task": task, "candidate": name})
                    mlflow.log_metrics(metrics)
                    if "weights" in candidates[name]:
                        mlflow.log_params({f"blend_weight.{k}": round(v, 3) for k, v in candidates[name]["weights"].items()})
                mlflow.log_metric(f"{name}.val_macro_f1", metrics["val_macro_f1"])

        champion_name = max(candidates, key=lambda n: candidates[n]["val"]["val_macro_f1"])
        champion = candidates[champion_name]
        model = champion["model"]
        logger.info("champion for %s: %s (%s)", task, champion_name, champion["val"])

        # -- memory members also remember the validation tickets (weights stay val-tuned)
        if config.get("refit_memory_on_train_val", True):
            if champion.get("refittable"):
                model = clone(model).fit(x_train + x_val, y_train + y_val)
            elif isinstance(model, BlendedTicketModel):
                for member_name in list(model.components):
                    if candidates[member_name].get("refittable"):
                        model.components[member_name] = clone(candidates[member_name]["model"]).fit(
                            x_train + x_val, y_train + y_val
                        )

        # -- the single test evaluation
        test_pred = model.predict(x_test)
        novelty_threshold = float(params.get("eval", {}).get("novelty_threshold", 0.6))
        try:
            novel = novelty_scores(x_train + x_val, x_test) < novelty_threshold
        except ValueError:  # vocabulary too small (toy data)
            novel = None
        test_metrics = classification_metrics(y_test, test_pred, labels=labels, novel_mask=novel)
        report = per_class_report(y_test, test_pred, labels)

        # -- persist the artifact DVC versions and the app serves
        model_file = save_model(model, output_dir / f"{task}_model.joblib")
        uses_transformer = "transformer" in getattr(model, "components", {}) or champion_name == "transformer"
        if not uses_transformer and (output_dir / "transformer").exists():
            import shutil

            shutil.rmtree(output_dir / "transformer")
        (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

        run_id = parent.info.run_id if parent else None
        payload = {
            "task": task,
            "split": "test",
            "model_type": champion_name,
            "model_path": relative_to_root(model_file),
            **{k: test_metrics[k] for k in test_metrics},
            "n_test": len(y_test),
            **champion["val"],
            "novelty_threshold": novelty_threshold,
            "blend_weights": champion.get("weights"),
            "candidates_val_macro_f1": {n: c["val"]["val_macro_f1"] for n, c in candidates.items()},
            "data_fingerprint": tags.get("data_fingerprint"),
            "git_commit": tags.get("git_commit"),
            "mlflow_run_id": run_id,
        }

        registry_version = None
        if track:
            mlflow.set_tag("champion_candidate", champion_name)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            _log_dict(report, f"evaluation/per_class_{task}.json")
            figure = confusion_figure(y_test, test_pred, labels, f"{task} – {champion_name} (test)")
            mlflow.log_figure(figure, f"evaluation/confusion_{task}.png")
            _log_model_to_mlflow(task, model_file, output_dir, x_test[:5])
            registry_version = _register(
                task,
                run_id,
                {**tags, "model_type": champion_name, "test_macro_f1": f"{test_metrics['macro_f1']:.4f}",
                 "test_novel_macro_f1": f"{test_metrics.get('novel_macro_f1', 0):.4f}"},
                f"{task} {champion_name}: test macro F1 {test_metrics['macro_f1']:.4f} "
                f"(novel tickets {test_metrics.get('novel_macro_f1', 0):.4f}), data {tags.get('data_fingerprint')}",
            )
            payload["registry_version"] = registry_version

        (output_dir / "champion.json").write_text(
            json.dumps({k: payload[k] for k in ("task", "model_type", "model_path", "macro_f1", "accuracy",
                                                 "novel_macro_f1", "n_test", "mlflow_run_id",
                                                 "data_fingerprint", "git_commit") if k in payload}
                       | {"registry_version": registry_version}, indent=2),
            encoding="utf-8",
        )
        _write_reports(task, payload, report, confusion_rows(y_test, test_pred), history, output_dir, is_real)
    finally:
        if parent is not None:
            mlflow.end_run()

    return {
        "task": task,
        "model_path": str(model_file),
        "model_type": champion_name,
        "macro_f1": test_metrics["macro_f1"],
        "novel_macro_f1": test_metrics.get("novel_macro_f1"),
        "accuracy": test_metrics["accuracy"],
        "mlflow_run_id": payload["mlflow_run_id"],
        "registry_version": payload.get("registry_version"),
        "candidates": {n: c["val"]["val_macro_f1"] for n, c in candidates.items()},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["queue", "priority"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-transformer", "--skip-distilbert", dest="skip", action="store_true")
    args = parser.parse_args()
    result = train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        task=args.task,
        config_path=args.config,
        use_transformer=False if args.skip else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
