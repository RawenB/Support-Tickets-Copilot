"""Baseline methods for incident alert comparison and trend analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STREAM_PATH = PROJECT_ROOT / "data" / "interim" / "incident_stream.csv"


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "incident.yaml"
    if not config_path.exists():
        return {"incident": {}}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _events_frame(events: list[dict[str, Any]] | pd.DataFrame | None) -> pd.DataFrame:
    if events is None:
        if not STREAM_PATH.exists():
            return pd.DataFrame(columns=["ticket_id", "queue", "priority", "created_at"])
        return pd.read_csv(STREAM_PATH)
    if isinstance(events, pd.DataFrame):
        return events.copy()
    return pd.DataFrame(events)


def baseline_alerts(
    events: list[dict[str, Any]] | pd.DataFrame | None = None,
    *,
    window_minutes: int | None = None,
    baseline_window: int | None = None,
    alert_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Return queue-level alerts using a rolling historical baseline."""
    config = _load_config().get("incident", {})
    window = int(
        window_minutes if window_minutes is not None else config.get("window_minutes", 60)
    )
    baseline_windows = int(
        baseline_window if baseline_window is not None else config.get("baseline_window", 7)
    )
    threshold = float(
        alert_threshold
        if alert_threshold is not None
        else config.get("alert_threshold", 0.80)
    )
    frame = _events_frame(events)
    if frame.empty or "created_at" not in frame.columns:
        return []
    frame = frame.copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["created_at"])
    if frame.empty:
        return []
    frame["queue"] = frame.get("queue", "general").fillna("general").astype(str).str.lower()
    end = frame["created_at"].max()
    current_start = end - pd.Timedelta(minutes=window)
    baseline_start = current_start - pd.Timedelta(minutes=window * baseline_windows)
    current = frame[frame["created_at"] > current_start]
    historical = frame[
        (frame["created_at"] > baseline_start) & (frame["created_at"] <= current_start)
    ]
    alerts: list[dict[str, Any]] = []
    for queue in sorted(frame["queue"].unique()):
        current_count = int((current["queue"] == queue).sum())
        history = historical[historical["queue"] == queue]
        if history.empty:
            baseline_mean = 0.0
            baseline_std = 0.0
        else:
            counts = history.set_index("created_at").resample(f"{window}min").size()
            baseline_mean = float(counts.mean()) if len(counts) else 0.0
            baseline_std = float(counts.std(ddof=0)) if len(counts) > 1 else 0.0
        ratio = current_count / max(baseline_mean, 1.0)
        z_score = (current_count - baseline_mean) / max(baseline_std, 1.0)
        alert = bool(
            current_count >= 2
            and (
                ratio >= max(1.5, threshold * 2)
                or z_score >= 2.0
            )
        )
        alerts.append(
            {
                "queue": queue,
                "current_count": current_count,
                "baseline_mean": round(baseline_mean, 3),
                "baseline_std": round(baseline_std, 3),
                "ratio": round(ratio, 3),
                "z_score": round(z_score, 3),
                "alert": alert,
            }
        )
    alerts.sort(key=lambda item: (item["alert"], item["ratio"]), reverse=True)
    return alerts
