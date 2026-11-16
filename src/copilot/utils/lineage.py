"""Reproducibility metadata (code version, data version, environment) for every tracked run."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def file_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprint(paths: list[Path]) -> str:
    """One short hash identifying the exact train/val/test files used (matches DVC content)."""
    digest = hashlib.md5()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(file_md5(path).encode())
    return digest.hexdigest()[:12]


def lineage_tags(data_paths: list[Path] | None = None) -> dict[str, str]:
    """Tags attached to MLflow runs and registry versions."""
    tags = {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))).lower(),
        "git_describe": _git("describe", "--tags", "--always", "--dirty") or "unknown",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if data_paths:
        existing = [path for path in data_paths if path.exists()]
        if existing:
            tags["data_fingerprint"] = data_fingerprint(existing)
            for path in existing:
                tags[f"data_md5_{path.stem}"] = file_md5(path)
    return tags
