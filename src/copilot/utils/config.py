"""Load and validate YAML configuration settings for support-copilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file; relative paths fall back to the project root."""
    resolved = Path(path)
    if not resolved.is_absolute() and not resolved.exists():
        resolved = PROJECT_ROOT / resolved
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
