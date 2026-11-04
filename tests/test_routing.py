from __future__ import annotations

from copilot.routing.review_queue import review_queue
from copilot.routing.router import route_ticket


def test_route_ticket_returns_queue_and_decision() -> None:
    result = route_ticket(
        {
            "subject": "Cannot sign in to my account",
            "body": "I keep getting a password reset loop",
            "priority": "high",
            "queue": "technical",
        },
        confidence=0.83,
        threshold=0.70,
    )
    assert result["queue"] == "technical"
    assert result["require_human_review"] is False


def test_review_queue_prioritizes_high_confidence_items() -> None:
    items = [
        {"ticket_id": "a", "priority": "low", "confidence": 0.25},
        {"ticket_id": "b", "priority": "high", "confidence": 0.85},
        {"ticket_id": "c", "priority": "medium", "confidence": 0.65},
    ]
    queue = review_queue(items, threshold=0.70)
    assert queue[0]["ticket_id"] == "b"
    assert len(queue) == 2
