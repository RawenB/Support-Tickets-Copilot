from __future__ import annotations

from app.services.ticket_service import (
    filter_tickets,
    format_ticket_age,
    get_active_tickets,
    get_dashboard_metrics,
    get_related_tickets,
    is_demo_mode,
    priority_distribution,
    queue_distribution,
    sort_tickets,
)
from copilot.db.repository import upsert_ticket


def test_service_returns_expected_fields() -> None:
    tickets = get_active_tickets(limit=3)
    assert tickets
    first = tickets[0]
    for key in ["ticket_id", "subject", "queue", "priority", "status", "customer_name"]:
        assert key in first


def test_demo_mode_flag_is_boolean() -> None:
    assert isinstance(is_demo_mode(), bool)


def test_live_sqlite_rows_take_priority_over_demo_fallback() -> None:
    ticket = {
        "ticket_id": "SIM-REAL-001",
        "subject": "Live queue update",
        "body": "This ticket is inserted through the live stream flow.",
        "queue": "technical",
        "priority": "high",
        "status": "new",
        "confidence": 0.88,
        "customer_name": "Live customer",
        "customer_email": "live@example.com",
        "assigned_agent": "Unassigned",
        "created_at": "2026-11-03T12:00:00+00:00",
    }
    upsert_ticket(ticket)
    rows = get_active_tickets()
    assert any(str(item["ticket_id"]) == "SIM-REAL-001" for item in rows)
    assert is_demo_mode() is False


def _sample_tickets() -> list[dict]:
    return [
        {
            "ticket_id": "1",
            "subject": "Login failure",
            "customer_name": "Alice",
            "customer_email": "alice@example.com",
            "queue": "technical",
            "priority": "critical",
            "status": "new",
            "confidence": 0.55,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "ticket_id": "2",
            "subject": "Invoice mismatch",
            "customer_name": "Marcus",
            "customer_email": "marcus@example.com",
            "queue": "billing",
            "priority": "medium",
            "status": "in_review",
            "confidence": 0.91,
        },
        {
            "ticket_id": "3",
            "subject": "Export question",
            "customer_name": "Priya",
            "customer_email": "priya@example.com",
            "queue": "technical",
            "priority": "low",
            "status": "escalated",
            "confidence": 0.65,
        },
    ]


def test_filter_tickets_applies_search_and_facets() -> None:
    tickets = _sample_tickets()
    assert [t["ticket_id"] for t in filter_tickets(tickets, search="alice")] == ["1"]
    assert [t["ticket_id"] for t in filter_tickets(tickets, queue="technical")] == ["1", "3"]
    assert [t["ticket_id"] for t in filter_tickets(tickets, status="escalated")] == ["3"]
    assert [t["ticket_id"] for t in filter_tickets(tickets, needs_review_only=True)] == [
        "1",
        "3",
    ]


def test_sort_tickets_orders_by_priority_and_confidence() -> None:
    tickets = _sample_tickets()
    assert [t["ticket_id"] for t in sort_tickets(tickets)] == ["1", "2", "3"]
    assert [t["ticket_id"] for t in sort_tickets(tickets, by="confidence")] == [
        "1",
        "3",
        "2",
    ]


def test_related_tickets_stay_within_queue() -> None:
    tickets = _sample_tickets()
    related = get_related_tickets(tickets[0], tickets)
    assert [t["ticket_id"] for t in related] == ["3"]


def test_dashboard_metrics_summarize_source() -> None:
    metrics = get_dashboard_metrics()
    assert metrics["total"] >= 1
    assert metrics["open"] <= metrics["total"]
    assert 0.0 <= float(metrics["avg_confidence"]) <= 1.0


def test_distribution_helpers_count_values() -> None:
    tickets = _sample_tickets()
    assert queue_distribution(tickets) == {"technical": 2, "billing": 1}
    assert priority_distribution(tickets) == {"critical": 1, "medium": 1, "low": 1}


def test_format_ticket_age_handles_missing_dates() -> None:
    assert format_ticket_age({}) == "recently"
    assert format_ticket_age({"created_at": "not-a-date"}) == "recently"
