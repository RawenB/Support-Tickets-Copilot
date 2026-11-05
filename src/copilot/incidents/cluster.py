"""Cluster incident events to detect anomalous support spikes."""

from __future__ import annotations

import json
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
            return pd.DataFrame(columns=["ticket_id", "queue", "priority", "created_at", "spike_id"])
        return pd.read_csv(STREAM_PATH)
    if isinstance(events, pd.DataFrame):
        return events.copy()
    return pd.DataFrame(events)


def cluster_events(
    events: list[dict[str, Any]] | pd.DataFrame | None = None,
    *,
    eps: float | None = None,
    min_samples: int | None = None,
    window_minutes: int | None = None,
) -> list[dict[str, Any]]:
    """Group incident-like events per queue using DBSCAN over event time."""
    config = _load_config().get("incident", {})
    eps_value = float(eps if eps is not None else config.get("eps", 0.3))
    min_samples_value = int(
        min_samples if min_samples is not None else config.get("min_samples", 5)
    )
    window_value = int(
        window_minutes
        if window_minutes is not None
        else config.get("window_minutes", 60)
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
    # Treat eps as a fraction of the monitoring window so the configured 0.3
    # clusters events within roughly 18 minutes of each other by default.
    eps_minutes = max(1.0, eps_value * window_value)
    from sklearn.cluster import DBSCAN

    clusters: list[dict[str, Any]] = []
    for queue, group in frame.groupby("queue"):
        ordered = group.sort_values("created_at").reset_index(drop=True)
        minutes = (
            ordered["created_at"] - ordered["created_at"].min()
        ).dt.total_seconds().to_numpy() / 60.0
        labels = DBSCAN(eps=eps_minutes, min_samples=min_samples_value).fit_predict(
            minutes.reshape(-1, 1)
        )
        for label in sorted(set(labels)):
            if label == -1:
                continue
            members = ordered[labels == label]
            if len(members) < min_samples_value:
                continue
            spike_ids = [
                str(value)
                for value in members.get("spike_id", pd.Series(dtype=str)).dropna().unique()
                if str(value).strip()
            ]
            clusters.append(
                {
                    "queue": queue,
                    "count": int(len(members)),
                    "start": members["created_at"].min().isoformat(),
                    "end": members["created_at"].max().isoformat(),
                    "ticket_ids": members["ticket_id"].astype(str).head(10).tolist()
                    if "ticket_id" in members
                    else [],
                    "spike_ids": spike_ids,
                    "high_priority": int(
                        members.get("priority", pd.Series(dtype=str))
                        .astype(str)
                        .str.lower()
                        .isin({"critical", "high"})
                        .sum()
                    ),
                }
            )
    clusters.sort(key=lambda item: item["count"], reverse=True)
    return clusters


def main() -> None:
    clusters = cluster_events()
    print(json.dumps({"n_clusters": len(clusters), "clusters": clusters}, indent=2))


if __name__ == "__main__":
    main()
