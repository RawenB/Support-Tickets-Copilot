"""Promote validated models to the production registry with governance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from copilot.utils.mlflow_utils import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _gate() -> float:
    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8")) or {}
    return float(params.get("f1_gate", 0.70))


def promote_model(
    task: str = "queue",
    *,
    run_id: str | None = None,
    alias: str = "champion",
    f1_gate: float | None = None,
) -> dict[str, Any]:
    """Promote the best run that passes the F1 gate to the requested alias."""
    metrics_path = PROJECT_ROOT / "reports" / f"metrics_{task}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file missing: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    gate = _gate() if f1_gate is None else float(f1_gate)
    if float(metrics.get("macro_f1", 0.0)) < gate:
        raise ValueError(
            f"{task} champion macro_f1={metrics.get('macro_f1')} is below gate {gate}"
        )

    selected_run = run_id or metrics.get("mlflow_run_id")
    if not selected_run:
        raise ValueError(f"{task} metrics do not contain an mlflow_run_id")
    client = setup_mlflow(f"support-copilot-{task}")
    name = f"support-copilot-{task}"
    try:
        client.create_registered_model(name)
    except Exception:
        pass
    source_run = selected_run
    try:
        version = client.create_model_version(
            name=name,
            source=f"runs:/{selected_run}/model",
            run_id=selected_run,
            description=f"Promoted {task} model ({metrics.get('model_type')}).",
        )
    except Exception:
        if metrics.get("model_type") != "distilbert":
            raise
        import mlflow
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path_value = metrics.get("model_path")
        if not model_path_value:
            champion_path = PROJECT_ROOT / "models" / f"{task}_model" / "champion.json"
            if champion_path.exists():
                champion = json.loads(champion_path.read_text(encoding="utf-8"))
                model_path_value = champion.get("model_path")
        model_path = Path(str(model_path_value or ""))
        model_dir = model_path.parent / "distilbert"
        if not model_dir.exists():
            raise
        with mlflow.start_run(run_name=f"{task}-promotion") as fallback_run:
            mlflow.transformers.log_model(
                transformers_model={
                    "model": AutoModelForSequenceClassification.from_pretrained(model_dir),
                    "tokenizer": AutoTokenizer.from_pretrained(model_dir),
                },
                artifact_path="model",
                task="text-classification",
            )
            source_run = fallback_run.info.run_id
        version = client.create_model_version(
            name=name,
            source=f"runs:/{source_run}/model",
            run_id=source_run,
            description=f"Promoted {task} DistilBERT model.",
        )
    client.set_registered_model_alias(name, alias, version.version)
    return {
        "task": task,
        "model_name": name,
        "version": str(version.version),
        "alias": alias,
        "run_id": source_run,
        "macro_f1": float(metrics["macro_f1"]),
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["queue", "priority"], required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()
    print(json.dumps(promote_model(args.task, run_id=args.run_id, alias=args.alias), indent=2))


if __name__ == "__main__":
    main()
