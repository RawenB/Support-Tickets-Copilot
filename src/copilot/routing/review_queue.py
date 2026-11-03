"""Queue review logic for human-in-the-loop support approval."""

from __future__ import annotations

from typing import Any, Iterable


def review_queue(
    items: Iterable[dict[str, Any]], *, threshold: float = 0.70
) -> list[dict[str, Any]]:
    """Return review items sorted by confidence and excluding low-priority noise."""
    filtered: list[dict[str, Any]] = []
    for item in items:
        priority = str(item.get("priority", "")).lower()
        if priority == "low":
            continue
        filtered.append(item)

    ordered = sorted(
        filtered,
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            1 if str(item.get("priority", "")).lower() == "high" else 0,
        ),
        reverse=True,
    )
    return ordered[:2]
