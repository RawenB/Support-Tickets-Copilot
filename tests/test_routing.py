from __future__ import annotations

from copilot.routing.review_queue import review_queue
from copilot.routing.router import route_ticket


def test_routing_modules_importable() -> None:
    assert callable(route_ticket)
    assert callable(review_queue)
