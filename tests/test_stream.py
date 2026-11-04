from __future__ import annotations

from datetime import datetime, timedelta

from copilot.incidents.simulate_stream import create_spike_cluster, insert_ticket_event


def test_simulator_inserts_ticket_event() -> None:
    record = insert_ticket_event(
        ticket_id="sim-001",
        subject="Password reset loop",
        body="I cannot sign in after reset.",
        queue="technical",
        priority="high",
        customer_email="user@example.com",
        language="en",
    )
    assert record["ticket_id"] == "sim-001"
    assert record["queue"] == "technical"
    assert record["created_at"]


def test_spike_injection_creates_cluster() -> None:
    cluster = create_spike_cluster(
        start_time=datetime.utcnow() - timedelta(minutes=3),
        count=5,
        queue="technical",
    )
    assert len(cluster) == 5
    assert all(item["queue"] == "technical" for item in cluster)
