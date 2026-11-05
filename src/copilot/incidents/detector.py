"""Detect operational incidents from support activity data streams."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from copilot.incidents.baseline import baseline_alerts
from copilot.incidents.cluster import cluster_events
from copilot.utils.mlflow_utils import setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STREAM_PATH = PROJECT_ROOT / "data" / "interim" / "incident_stream.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "incident_metrics.json"


def _load_events(events: list[dict[str, Any]] | pd.DataFrame | None) -> pd.DataFrame:
    if events is not None:
        return events.copy() if isinstance(events, pd.DataFrame) else pd.DataFrame(events)
    if not STREAM_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(STREAM_PATH)


def detect_incidents(
    events: list[dict[str, Any]] | pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Raise alert candidates when support activity deviates from expected patterns."""
    frame = _load_events(events)
    if frame.empty:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_events": 0,
            "n_clusters": 0,
            "n_baseline_alerts": 0,
            "detected_spike_ids": [],
            "labelled_spike_ids": [],
            "spike_precision": 0.0,
            "spike_recall": 0.0,
            "clusters": [],
            "baseline": [],
        }
    else:
        clusters = cluster_events(frame)
        baseline = baseline_alerts(frame)
        labelled = (
            {str(value) for value in frame["spike_id"].dropna().unique() if str(value).strip()}
            if "spike_id" in frame.columns
            else set()
        )
        detected = {
            spike_id
            for cluster in clusters
            for spike_id in cluster.get("spike_ids", [])
            if spike_id
        }
        true_positives = len(detected & labelled)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_events": int(len(frame)),
            "n_clusters": len(clusters),
            "n_baseline_alerts": sum(1 for item in baseline if item["alert"]),
            "detected_spike_ids": sorted(detected),
            "labelled_spike_ids": sorted(labelled),
            "spike_precision": round(true_positives / len(detected), 4) if detected else 0.0,
            "spike_recall": round(true_positives / len(labelled), 4) if labelled else 0.0,
            "clusters": clusters,
            "baseline": baseline,
        }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    setup_mlflow("support-copilot-incidents")
    with mlflow.start_run(run_name="incident-eval"):
        mlflow.log_metrics(
            {
                "n_events": payload["n_events"],
                "n_clusters": payload["n_clusters"],
                "n_baseline_alerts": payload["n_baseline_alerts"],
                "spike_precision": payload["spike_precision"],
                "spike_recall": payload["spike_recall"],
            }
        )
        mlflow.log_artifact(str(REPORT_PATH))
    return payload


def main() -> None:
    print(json.dumps(detect_incidents(), indent=2))


if __name__ == "__main__":
    main()
