"""Service helpers for live ticket stream replay and incident simulation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from copilot.incidents.simulate_stream import create_spike_cluster, insert_ticket_event


def replay_stream(limit: int = 6) -> list[dict[str, Any]]:
    """Replay the English test split into SQLite at a low cadence for a live inbox."""
    from copilot.incidents.simulate_stream import simulate_stream

    return simulate_stream(limit=limit)


def trigger_spike(queue: str = "technical_support") -> list[dict[str, Any]]:
    """Trigger a synthetic incident burst for admin testing."""
    start_time = datetime.now(timezone.utc) - timedelta(minutes=3)
    return create_spike_cluster(start_time=start_time, count=30, queue=queue)


def insert_demo_tick(
    ticket_id: str, *, subject: str, body: str, queue: str, priority: str
) -> dict[str, Any]:
    """Insert a single demo ticket and return the persisted record."""
    return insert_ticket_event(
        ticket_id=ticket_id,
        subject=subject,
        body=body,
        queue=queue,
        priority=priority,
        customer_email=f"{ticket_id.lower()}@example.com",
        language="en",
    )
