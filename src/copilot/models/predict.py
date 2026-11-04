"""Inference utilities for queue and priority prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


def predict(
    text: str | list[str],
    *,
    task: str = "queue",
    model_path: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict a queue or priority label and confidence for one or many text payloads."""
    if model_path is None:
        model_path = Path("models") / f"{task}_baseline_model" / f"{task}_baseline_model.joblib"
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {resolved_path}")

    model = joblib.load(resolved_path)
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
                "model_path": str(resolved_path),
            }
        )

    if isinstance(text, str):
        return outputs[0]
    return outputs
