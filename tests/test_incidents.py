from __future__ import annotations

from copilot.incidents.baseline import baseline_alerts
from copilot.incidents.cluster import cluster_events
from copilot.incidents.detector import detect_incidents
from copilot.incidents.simulate_stream import simulate_stream


def test_incidents_modules_importable() -> None:
    assert callable(cluster_events)
    assert callable(baseline_alerts)
    assert callable(detect_incidents)
    assert callable(simulate_stream)


def _spike_frame():
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    rows = [
        {
            "ticket_id": f"T-{index}",
            "queue": "technical_support",
            "priority": "high" if index < 5 else "medium",
            "created_at": (start + timedelta(minutes=index * 2)).isoformat(),
            "spike_id": "SPIKE-1",
        }
        for index in range(12)
    ]
    rows.append(
        {
            "ticket_id": "T-normal",
            "queue": "billing_and_payments",
            "priority": "low",
            "created_at": (start + timedelta(hours=4)).isoformat(),
            "spike_id": None,
        }
    )
    return pd.DataFrame(rows)


def test_cluster_events_finds_real_spike_cluster() -> None:
    clusters = cluster_events(_spike_frame())
    assert clusters
    assert clusters[0]["queue"] == "technical_support"
    assert clusters[0]["count"] == 12
    assert clusters[0]["spike_ids"] == ["SPIKE-1"]


def test_baseline_alerts_returns_queue_comparison() -> None:
    alerts = baseline_alerts(_spike_frame())
    assert alerts
    assert {"queue", "current_count", "baseline_mean", "alert"} <= set(alerts[0])


def test_detector_writes_real_incident_metrics(tmp_path, monkeypatch) -> None:
    import copilot.incidents.detector as detector

    report = tmp_path / "incident_metrics.json"
    monkeypatch.setattr(detector, "REPORT_PATH", report)
    payload = detector.detect_incidents(_spike_frame())
    assert report.exists()
    assert payload["n_events"] == 13
    assert payload["n_clusters"] >= 1
    assert payload["spike_recall"] == 1.0
