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
