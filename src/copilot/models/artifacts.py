"""Load and save model artifacts in one place, so path handling stays portable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_model(path: str | Path) -> Any:
    """Load a pickled ticket model and bind any on-disk weights to the pickle's folder."""
    resolved = Path(path)
    model = joblib.load(resolved)
    if hasattr(model, "bind"):
        model.bind(resolved.parent)
    return model


def save_model(model: Any, path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, resolved)
    return resolved


def relative_to_root(path: str | Path) -> str:
    """Store project paths relative to the repo root, never as machine-specific absolutes."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def resolve_from_root(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
