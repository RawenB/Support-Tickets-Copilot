"""Generate a first-draft answer using retrieved support context."""

from __future__ import annotations

from typing import Any


def generate_answer(
    ticket: dict[str, Any] | str | None = None,
    *,
    passages: list[str] | None = None,
) -> str:
    """Generate a grounded draft response to the support ticket."""
    if isinstance(ticket, dict):
        subject = str(ticket.get("subject") or "support request")
        queue = str(ticket.get("queue") or "general")
        body = str(ticket.get("body") or "Customer reported an issue.")
        context = " ".join(passages or [])
        if context:
            return (
                f"Thanks for reporting the {queue} issue about {subject}. "
                f"Based on the current guidance, {context[:240]} "
                "I’m confirming the next step and will keep the update linked "
                f"to the original issue: {body[:180]}."
            )
        return (
            f"Thanks for reporting the {queue} issue about {subject}. "
            "I’ve reviewed the current context and am confirming the next "
            "step before sending the final update. "
            f"Customer details: {body[:180]}."
        )
    if isinstance(ticket, str):
        return (
            "Thanks for reporting this issue. I’ve reviewed the latest "
            f"context and am confirming the next step for: {ticket[:180]}."
        )
    return (
        "Thanks for reporting this issue. I’ve reviewed the current context "
        "and am confirming the next step with the right team before sending "
        "the final update."
    )
