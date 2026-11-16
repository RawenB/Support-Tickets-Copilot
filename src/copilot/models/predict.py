"""Inference utilities for queue and priority prediction."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from copilot.data.text import build_text
from copilot.models.artifacts import load_model, resolve_from_root
from copilot.utils.mlflow_utils import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _local_champion_path(task: str) -> Path | None:
    champion = PROJECT_ROOT / "models" / f"{task}_model" / "champion.json"
    if champion.exists():
        try:
            payload = json.loads(champion.read_text(encoding="utf-8"))
            path = resolve_from_root(payload.get("model_path", ""))
            if path.is_file():
                return path
        except Exception:
            pass
    path = PROJECT_ROOT / "models" / f"{task}_model" / f"{task}_model.joblib"
    return path if path.exists() else None


def _load_registry_champion(task: str) -> Any | None:
    """Load ``models:/support-copilot-<task>@champion`` from the MLflow registry."""
    try:
        setup_mlflow(f"support-copilot-{task}")
        uri = f"models:/support-copilot-{task}@champion"
        try:
            pyfunc_model = mlflow.pyfunc.load_model(uri)
            inner = pyfunc_model.unwrap_python_model()
            return getattr(inner, "model", None) or mlflow.sklearn.load_model(uri)
        except Exception:
            return mlflow.sklearn.load_model(uri)
    except Exception:
        return None


_CACHE: dict[tuple[str, str], tuple[Any, float]] = {}
_LOCK = threading.Lock()


def _cached(path: Path) -> Any:
    """Load once per process; reload automatically when the artifact file changes."""
    key = ("file", str(path))
    mtime = path.stat().st_mtime
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is None or hit[1] != mtime:
            _CACHE[key] = (load_model(path), mtime)
        return _CACHE[key][0]


def clear_model_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _resolve_model(task: str, model_path: str | Path | None) -> tuple[Any, str]:
    if model_path is not None:
        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model artifact not found: {resolved}")
        return _cached(resolved), str(resolved)
    local = _local_champion_path(task)
    if local is not None:
        return _cached(local), str(local)
    registry_model = _load_registry_champion(task)
    if registry_model is not None:
        return registry_model, f"models:/support-copilot-{task}@champion"
    # No silent fallback to a toy model: callers store "unclassified" instead.
    raise FileNotFoundError(f"No champion model artifact found for task '{task}'.")


def predict(
    text: str | list[str],
    *,
    task: str = "queue",
    model_path: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict a queue or priority label and confidence for one or many text payloads."""
    model, resolved_path = _resolve_model(task, model_path)
    raw = [text] if isinstance(text, str) else list(text)
    inputs = [build_text("", item) for item in raw]  # same normalisation as training
    probabilities = np.asarray(model.predict_proba(inputs))
    classes = np.asarray(model.classes_)
    outputs: list[dict[str, Any]] = []
    for probs in probabilities:
        outputs.append(
            {
                "task": task,
                "label": str(classes[int(np.argmax(probs))]),
                "confidence": round(float(np.max(probs)), 4),
                "model_path": resolved_path,
            }
        )

    if isinstance(text, str):
        return outputs[0]
    return outputs
