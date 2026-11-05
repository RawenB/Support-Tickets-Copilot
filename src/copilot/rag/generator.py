"""Generate a first-draft answer using retrieved support context."""

from __future__ import annotations

import re
from typing import Any

STYLES = ("concise", "detailed", "friendly")


def _first_name(ticket: dict[str, Any]) -> str:
    name = str(ticket.get("customer_name") or "").strip()
    if not name or name.lower() in {"unknown customer", "test customer"}:
        return "there"
    return name.split()[0].title()


def _sentences(text: str, limit: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts[:limit]


def _guidance(passages: list[str], variant: int) -> str:
    """Rotate through the retrieved passages so each regeneration can use different evidence."""
    if not passages:
        return ""
    offset = variant % len(passages)
    ordered = passages[offset:] + passages[:offset]
    return " ".join(_sentences(ordered[0], 2))


def generate_answer(
    ticket: dict[str, Any] | str | None = None,
    *,
    passages: list[str] | None = None,
    variant: int = 0,
) -> str:
    """Generate a grounded draft response; `variant` selects the style (0, 1, 2, then repeats)."""
    if isinstance(ticket, str):
        return (
            "Thanks for reporting this issue. I have reviewed the latest "
            f"context and am confirming the next step for: {ticket[:180]}."
        )
    if not isinstance(ticket, dict):
        return (
            "Thanks for reporting this issue. I have reviewed the current context "
            "and am confirming the next step with the right team before sending "
            "the final update."
        )

    subject = str(ticket.get("subject") or "your request").strip()
    queue = str(ticket.get("queue") or "support").replace("_", " ").strip()
    body = " ".join(_sentences(str(ticket.get("body") or ""), 1))[:200]
    name = _first_name(ticket)
    guidance = _guidance(list(passages or []), variant)
    style = STYLES[variant % len(STYLES)]

    if style == "concise":
        text = f"Hi {name}, thanks for contacting us about \"{subject}\". "
        text += guidance + " " if guidance else "We are looking into it now. "
        return text + "We will update you shortly.\n\nBest regards,\nSupport Team"

    if style == "detailed":
        lines = [f"Hello {name},", "", f"Thank you for reaching out about \"{subject}\" ({queue})."]
        if body:
            lines.append(f"From your message, we understand: {body}")
        lines.append("")
        lines.append("Here is what we recommend:")
        if guidance:
            lines.append(f"1. {guidance}")
            lines.append("2. If the problem continues, reply with the exact error message and the time it happened.")
        else:
            lines.append("1. Please send us the exact error message and the time it happened.")
            lines.append("2. We will check the account and confirm the next step.")
        lines += ["", "We will keep this ticket open until it is resolved.", "", "Kind regards,", "Support Team"]
        return "\n".join(lines)

    text = f"Hi {name}, sorry for the trouble with \"{subject}\" - we know how frustrating that is. "
    text += (guidance + " ") if guidance else "I am checking this with the right team right now. "
    return text + "Please reply here if anything changes, and I will stay on it until it is fixed.\n\nWarm regards,\nYour Support Team"
