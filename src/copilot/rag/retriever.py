"""Retrieve top support documents for a ticket via vector search."""

from __future__ import annotations

from typing import Any

KEYWORDS = (
    "login",
    "access",
    "password",
    "billing",
    "charge",
    "invoice",
    "reset",
    "account",
    "plan",
)

FAQ_PASSAGES = [
    "If a user cannot sign in after a password reset, verify that the reset "
    "flow completed, the session is expired, and the account has not been "
    "locked by a security policy.",
    "For duplicate billing charges, confirm whether the invoice was retried or "
    "generated twice, then review the payment provider logs and issue a credit "
    "if the duplicate is confirmed.",
    "For access or approval issues, review recent team role changes, permission "
    "sync delays, and policy updates before sending a final resolution.",
    "If a subscription or account change is requested, confirm the plan tiers, "
    "recent billing events, and any customer consent before updating the "
    "account.",
]


def retrieve(
    ticket: dict[str, Any] | str | None = None,
    *,
    limit: int = 3,
) -> list[str]:
    """Return the most relevant FAQ context for a ticket or issue description."""
    if isinstance(ticket, str):
        text = ticket.lower()
        matches = [
            passage
            for passage in FAQ_PASSAGES
            if any(keyword in text for keyword in KEYWORDS)
        ]
        return matches[: max(1, int(limit))]

    if isinstance(ticket, dict):
        haystack = " ".join(
            str(ticket.get(key, ""))
            for key in ("subject", "body", "queue", "priority")
        ).lower()
        weighted: list[tuple[int, str]] = []
        for passage in FAQ_PASSAGES:
            score = 0
            for keyword in KEYWORDS:
                if keyword in passage.lower() and keyword in haystack:
                    score += 2
                if keyword in passage.lower():
                    score += 1
            weighted.append((score, passage))
        ranked = [passage for _, passage in sorted(weighted, reverse=True)]
        if not ranked:
            return FAQ_PASSAGES[: max(1, int(limit))]
        return ranked[: max(1, int(limit))]

    return FAQ_PASSAGES[: max(1, int(limit))]
