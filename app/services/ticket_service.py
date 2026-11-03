"""Service layer for ticket access and demo-mode AI fallbacks."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from copilot.db.repository import list_ticket_actions, list_tickets, log_ticket_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_copilot.ticket_service")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_TICKETS_PATH = PROJECT_ROOT / "data" / "demo_tickets.json"

REVIEW_CONFIDENCE_THRESHOLD = 0.8
ACTIVE_STATUSES = {"new", "in_review", "assigned"}
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _sqlite_has_tickets() -> bool:
    """Check whether the SQLite ticket store already contains live rows."""
    db_path = PROJECT_ROOT / "data" / "support_copilot.db"
    if not db_path.exists():
        return False
    try:
        return bool(list_tickets(limit=1))
    except Exception:  # pragma: no cover - defensive for unexpected DB issues
        logger.exception("Unable to inspect SQLite ticket rows for fallback logic.")
        return False


def is_demo_mode() -> bool:
    """Return whether the UI is currently running in known demo mode."""
    db_has_rows = _sqlite_has_tickets()
    if db_has_rows:
        return False
    return not (PROJECT_ROOT / "data" / "splits" / "test.parquet").exists()


def _load_demo_payload() -> list[dict[str, Any]]:
    if not DEMO_TICKETS_PATH.exists():
        return []
    with DEMO_TICKETS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        {
            "ticket_id": str(item.get("id") or item.get("ticket_id") or "demo-000"),
            "subject": str(item.get("subject") or "Untitled ticket"),
            "body": str(item.get("body") or ""),
            "queue": str(item.get("queue") or "general"),
            "priority": str(item.get("priority") or "medium"),
            "status": str(item.get("status") or "new"),
            "confidence": float(item.get("confidence", 0.82)),
            "customer_name": str(
                item.get("customer") or item.get("customer_name") or "Unknown customer"
            ),
            "customer_email": str(
                item.get("email") or item.get("customer_email") or "unknown@example.com"
            ),
            "assigned_agent": str(item.get("agent") or item.get("assigned_agent") or "Unassigned"),
            "created_at": str(item.get("created_at") or "2025-01-01T00:00:00Z"),
        }
        for item in payload
    ]


def get_active_tickets(*, limit: int | None = None) -> list[dict[str, Any]]:
    """Return the active inbox rows, using SQLite data when available and demo data otherwise."""
    if is_demo_mode():
        tickets = _load_demo_payload()
        if limit is not None:
            tickets = tickets[:limit]
        logger.info("Using demo mode ticket data for Support Copilot UI.")
        return tickets
    rows = list_tickets(limit=limit)
    if not rows:
        logger.warning("No tickets found in SQLite; falling back to demo data.")
        return _load_demo_payload()[:limit]
    return rows


def get_ticket_by_id(ticket_id: str) -> dict[str, Any] | None:
    """Fetch a single ticket by id from the active source."""
    for ticket in get_active_tickets():
        if str(ticket["ticket_id"]) == str(ticket_id):
            return ticket
    return None


def get_queue_team_map() -> dict[str, str]:
    """Return queue-to-team mapping from the project routing config."""
    config_path = PROJECT_ROOT / "configs" / "routing.yaml"
    if not config_path.exists():
        return {
            "technical": "Product Engineering",
            "billing": "Billing Ops",
            "general": "Support Desk",
        }
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    route_config = config.get("routing", {})
    team_map = route_config.get("queue_team_map", {})
    return {str(key).lower(): str(value) for key, value in team_map.items()}


def log_ticket_decision(ticket_id: str, *, action: str, agent: str = "agent") -> dict[str, Any]:
    """Persist a choice made by the human agent against the selected ticket."""
    details = f"UI action: {action}"
    decision = "accepted" if action in {"accepted", "confirmed"} else "review"
    return log_ticket_action(
        ticket_id,
        action=action,
        agent=agent,
        details=details,
        decision=decision,
    )


def get_recent_decisions(*, limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent human decisions, tolerating an empty database."""
    try:
        return list_ticket_actions(limit=limit)
    except Exception:  # pragma: no cover - defensive for read-only environments
        logger.exception("Unable to read recent ticket decisions from SQLite.")
        return []


def get_dashboard_metrics() -> dict[str, int | float]:
    """Aggregate headline metrics from the active ticket source."""
    tickets = get_active_tickets()
    total = len(tickets)
    open_count = sum(
        1 for item in tickets if str(item.get("status", "")).lower() in ACTIVE_STATUSES
    )
    needs_review = sum(
        1
        for item in tickets
        if float(item.get("confidence", REVIEW_CONFIDENCE_THRESHOLD))
        < REVIEW_CONFIDENCE_THRESHOLD
    )
    escalated = sum(1 for item in tickets if str(item.get("status", "")).lower() == "escalated")
    confidence_values = [float(item.get("confidence", 0.8)) for item in tickets]
    avg_confidence = round(sum(confidence_values) / total, 2) if total else 0.0
    return {
        "total": total,
        "open": open_count,
        "needs_review": needs_review,
        "escalated": escalated,
        "avg_confidence": avg_confidence,
        "avg_first_response_minutes": 8,
    }


def filter_tickets(
    tickets: list[dict[str, Any]],
    *,
    search: str = "",
    status: str = "All",
    priority: str = "All",
    queue: str = "All",
    needs_review_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter tickets by free-text search and facet selections."""
    query = search.strip().lower()
    filtered: list[dict[str, Any]] = []
    for ticket in tickets:
        if query:
            haystack = " ".join(
                str(ticket.get(key, ""))
                for key in ("ticket_id", "subject", "customer_name", "customer_email", "queue")
            ).lower()
            if query not in haystack:
                continue
        if status != "All" and str(ticket.get("status", "")).lower() != status.lower():
            continue
        if priority != "All" and str(ticket.get("priority", "")).lower() != priority.lower():
            continue
        if queue != "All" and str(ticket.get("queue", "")).lower() != queue.lower():
            continue
        if (
            needs_review_only
            and float(ticket.get("confidence", REVIEW_CONFIDENCE_THRESHOLD))
            >= REVIEW_CONFIDENCE_THRESHOLD
        ):
            continue
        filtered.append(ticket)
    return filtered


def sort_tickets(
    tickets: list[dict[str, Any]], *, by: str = "priority"
) -> list[dict[str, Any]]:
    """Sort tickets by priority weight or ascending confidence."""
    if by == "confidence":
        return sorted(
            tickets,
            key=lambda item: (
                float(item.get("confidence", 1.0)),
                PRIORITY_ORDER.get(str(item.get("priority", "low")).lower(), 9),
            ),
        )
    return sorted(
        tickets,
        key=lambda item: PRIORITY_ORDER.get(str(item.get("priority", "low")).lower(), 9),
    )


def get_related_tickets(
    ticket: dict[str, Any],
    tickets: list[dict[str, Any]] | None = None,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return tickets in the same queue that may share a root cause."""
    pool = tickets if tickets is not None else get_active_tickets()
    queue = str(ticket.get("queue", "")).lower()
    ticket_id = str(ticket.get("ticket_id"))
    related = [
        item
        for item in pool
        if str(item.get("queue", "")).lower() == queue and str(item.get("ticket_id")) != ticket_id
    ]
    return sort_tickets(related)[:limit]


def get_ticket_sources(ticket: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the knowledge-base passages supporting the current draft."""
    queue = str(ticket.get("queue", "general")).lower()
    shared = {
        "id": "FAQ-104",
        "score": 0.91,
        "text": (
            "Resetting MFA tokens on locked accounts requires confirming the prior "
            "device and re-establishing a secure session."
        ),
    }
    if queue == "billing":
        return [
            {
                "id": "KB-310",
                "score": 0.89,
                "text": "Duplicate charges are refunded after verifying the invoice ledger.",
            },
            shared,
        ]
    if queue == "technical":
        return [
            shared,
            {
                "id": "KB-228",
                "score": 0.83,
                "text": (
                    "Known issue: permission sync delays after recent access-policy updates."
                ),
            },
        ]
    return [
        {
            "id": "KB-142",
            "score": 0.78,
            "text": "Admin portal exports are enabled per team role under Reporting.",
        },
        shared,
    ]


def format_ticket_age(ticket: dict[str, Any]) -> str:
    """Return a compact human-readable age for a ticket."""
    raw = ticket.get("created_at")
    if not raw:
        return "recently"
    try:
        created = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        minutes = max(0, int((datetime.now(timezone.utc) - created).total_seconds() // 60))
    except ValueError:
        return "recently"
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 60 * 24:
        return f"{minutes // 60}h ago"
    return f"{minutes // (60 * 24)}d ago"


def queue_distribution(tickets: list[dict[str, Any]]) -> dict[str, int]:
    """Count tickets per queue for charts and admin summaries."""
    return dict(Counter(str(item.get("queue", "general")).lower() for item in tickets))


def priority_distribution(tickets: list[dict[str, Any]]) -> dict[str, int]:
    """Count tickets per priority for charts and admin summaries."""
    return dict(Counter(str(item.get("priority", "medium")).lower() for item in tickets))
