"""Route incoming tickets to queues and review flows based on model confidence."""

from __future__ import annotations

from typing import Any


def route_ticket(
    ticket: dict[str, Any],
    *,
    confidence: float,
    threshold: float = 0.70,
    default_queue: str = "manual_review",
) -> dict[str, Any]:
    """Return a routing decision with an explicit human-review flag."""
    queue = str(ticket.get("queue") or ticket.get("predicted_queue") or "general")
    is_review_required = float(confidence) < float(threshold)

    if is_review_required:
        queue = default_queue

    return {
        "ticket_id": ticket.get("ticket_id"),
        "queue": queue,
        "priority": ticket.get("priority"),
        "confidence": float(confidence),
        "require_human_review": is_review_required,
    }
