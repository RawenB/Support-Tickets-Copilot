"""The real dataset has 5 priority levels and a configurable review threshold."""
from __future__ import annotations

from app.services import ticket_service as svc
from copilot.db import repository as repo

REAL_PRIORITIES = ["critical", "high", "medium", "low", "very_low"]


def test_priority_scale_includes_very_low():
    assert "very_low" in svc.PRIORITY_ORDER
    assert svc.PRIORITY_ORDER["critical"] < svc.PRIORITY_ORDER["very_low"]
    assert "very_low" in svc.get_priority_choices()


def test_inbox_query_orders_critical_first(make_ticket):
    for i, priority in enumerate(reversed(REAL_PRIORITIES)):
        make_ticket(f"P-{i}", priority=priority, created_at="2026-11-04T10:00:00+00:00")
    order = [t["priority"] for t in repo.list_tickets()]
    assert order == REAL_PRIORITIES, f"expected critical->very_low, got {order}"


def test_review_threshold_follows_config(monkeypatch):
    monkeypatch.setattr(
        svc, "_load_routing_config", lambda: {"routing": {"review_threshold": 0.70}}
    )
    assert svc.get_review_threshold() == 0.70


def test_queue_choices_come_from_config_not_hardcoded_demo_queues():
    choices = {c.lower() for c in svc.get_queue_choices()}
    assert choices, "queue choices must not be empty"
    assert not choices <= {"technical", "billing", "general"}, "still the demo queues"
