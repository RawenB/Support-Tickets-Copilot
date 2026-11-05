"""Inference utilities for queue and priority prediction."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np

from copilot.utils.mlflow_utils import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _local_champion_path(task: str) -> Path | None:
    champion = PROJECT_ROOT / "models" / f"{task}_model" / "champion.json"
    if champion.exists():
        try:
            import json

            payload = json.loads(champion.read_text(encoding="utf-8"))
            path = Path(payload.get("model_path", ""))
            if path.exists():
                return path
        except Exception:
            pass
    path = PROJECT_ROOT / "models" / f"{task}_model" / f"{task}_model.joblib"
    return path if path.exists() else None


def _load_registry_champion(task: str) -> Any | None:
    try:
        client = setup_mlflow(f"support-copilot-{task}")
        version = client.get_model_version_by_alias(f"support-copilot-{task}", "champion")
        if version is None:
            return None
        run = client.get_run(version.run_id)
        model_type = run.data.params.get("model_type", "")
        if model_type == "distilbert":
            destination = tempfile.mkdtemp(prefix=f"{task}-champion-")
            local_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{version.run_id}/model",
                dst_path=destination,
            )
            candidates = list(Path(local_dir).rglob("*.joblib"))
            if not candidates:
                return None
            return joblib.load(candidates[0])
        return mlflow.sklearn.load_model(f"models:/support-copilot-{task}@champion")
    except Exception:
        return None


def _resolve_model(task: str, model_path: str | Path | None) -> tuple[Any, str]:
    if model_path is not None:
        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model artifact not found: {resolved}")
        return joblib.load(resolved), str(resolved)
    local = _local_champion_path(task)
    if local is not None:
        return joblib.load(local), str(local)
    registry_model = _load_registry_champion(task)
    if registry_model is not None:
        return registry_model, f"models:/support-copilot-{task}@champion"
    baseline = PROJECT_ROOT / "models" / f"{task}_baseline_model" / f"{task}_baseline_model.joblib"
    if not baseline.exists():
        baseline = PROJECT_ROOT / "models" / f"{task}_baseline_model.joblib"
    if baseline.exists():
        return joblib.load(baseline), str(baseline)
    raise FileNotFoundError(
        f"No champion or baseline model artifact found for task '{task}'."
    )


def predict(
    text: str | list[str],
    *,
    task: str = "queue",
    model_path: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict a queue or priority label and confidence for one or many text payloads."""
    model, resolved_path = _resolve_model(task, model_path)
    inputs = [text] if isinstance(text, str) else list(text)
    predictions = model.predict(inputs)
    probabilities = model.predict_proba(inputs)
    outputs: list[dict[str, Any]] = []
    for index, label in enumerate(predictions):
        probs = probabilities[index]
        outputs.append(
            {
                "task": task,
                "label": str(label),
                "confidence": round(float(np.max(probs)), 4),
                "model_path": resolved_path,
            }
        )

    if isinstance(text, str):
        return outputs[0]
    return outputs
