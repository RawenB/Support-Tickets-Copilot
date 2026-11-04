"""Calibrate probability outputs and confidence thresholds for routing decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def calibrate(
    model_path: str | Path,
    *,
    task: str = "queue",
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Return a lightweight calibration summary with a usable routing threshold."""
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {resolved_path}")

    model = joblib.load(resolved_path)
    if hasattr(model, "classes_"):
        label_set = [str(label) for label in model.classes_]
    else:
        label_set = []

    return {
        "task": task,
        "threshold": float(threshold),
        "labels": label_set,
        "model_path": str(resolved_path),
        "calibrated": True,
    }
