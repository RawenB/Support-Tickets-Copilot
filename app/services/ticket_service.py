"""Service layer for ticket access, draft history, and safe review actions."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from copilot.db.repository import (
    add_correction,
    add_outcome,
    get_latest_draft,
    list_drafts,
    list_ticket_actions,
    list_tickets,
    log_ticket_action,
    set_ticket_status,
    update_ticket_fields,
)
from copilot.db.repository import (
    save_draft as repo_save_draft,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_copilot.ticket_service")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_TICKETS_PATH = PROJECT_ROOT / "data" / "demo_tickets.json"
ACTIVE_STATUSES = {"new", "in_review", "assigned"}
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "very_low": 4}


@dataclass
class ActionResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def _load_routing_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "routing.yaml"
    if not config_path.exists():
        return {
            "routing": {
                "review_threshold": 0.7,
                "queue_team_map": {
                    "technical": "Product Engineering",
                    "billing": "Billing Ops",
                    "general": "Support Desk",
                },
                "priorities": ["critical", "high", "medium", "low", "very_low"],
            }
        }
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _load_params_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "params.yaml"
    if not config_path.exists():
        return {"review_threshold": 0.70}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def get_review_threshold() -> float:
    config = _load_routing_config().get("routing", {})
    threshold = config.get("review_threshold")
    if threshold is None:
        threshold = _load_params_config().get("review_threshold", 0.70)
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return 0.70


def get_queue_choices() -> list[str]:
    config = _load_routing_config().get("routing", {})
    queue_map = config.get("queue_team_map", {})
    if isinstance(queue_map, dict) and queue_map:
        return [str(key).title() for key in queue_map.keys()]
    return ["Technical", "Billing", "General"]


def get_priority_choices() -> list[str]:
    config = _load_routing_config().get("routing", {})
    priorities = config.get("priorities", ["critical", "high", "medium", "low", "very_low"])
    return [str(item).lower() for item in priorities]


def _sqlite_has_tickets() -> bool:
    """Check whether the SQLite ticket store already contains live rows."""
    db_path = PROJECT_ROOT / "data" / "support_copilot.db"
    if not db_path.exists():
        return False
    try:
        return bool(list_tickets(limit=1))
    except Exception:  # pragma: no cover
        logger.exception("Unable to inspect SQLite ticket rows for fallback logic.")
        return False


def is_demo_mode() -> bool:
    """Return whether the UI is currently running in known demo mode."""
    if _sqlite_has_tickets():
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
            "team": str(item.get("team") or "Support Desk"),
            "created_at": str(item.get("created_at") or "2025-01-01T00:00:00Z"),
        }
        for item in payload
    ]


def get_active_tickets(*, limit: int | None = None) -> list[dict[str, Any]]:
    """Return the active inbox rows, preferring SQLite data when present."""
    try:
        rows = list_tickets(limit=limit)
    except Exception:  # pragma: no cover
        rows = []
    if rows:
        return rows
    tickets = _load_demo_payload()
    if limit is not None:
        tickets = tickets[:limit]
    logger.info("Using demo mode ticket data for Support Copilot UI.")
    return tickets


def get_ticket_by_id(ticket_id: str) -> dict[str, Any] | None:
    """Fetch a single ticket by id from the active source."""
    for ticket in get_active_tickets():
        if str(ticket["ticket_id"]) == str(ticket_id):
            return ticket
    return None


def get_queue_team_map() -> dict[str, str]:
    """Return queue-to-team mapping from the project routing config."""
    config = _load_routing_config().get("routing", {})
    team_map = config.get("queue_team_map", {})
    return {str(key).lower(): str(value) for key, value in team_map.items()}


def get_recent_decisions(*, limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent human decisions, tolerating an empty database."""
    try:
        return list_ticket_actions(limit=limit)
    except Exception:  # pragma: no cover
        logger.exception("Unable to read recent ticket decisions from SQLite.")
        return []


def _first_response_minutes_for_ticket(
    ticket: dict[str, Any],
    actions: list[dict[str, Any]],
) -> int | None:
    ticket_id = str(ticket.get("ticket_id"))
    created_at = _parse_datetime(ticket.get("created_at"))
    if created_at is None:
        return None
    relevant = []
    for action in actions:
        if str(action.get("ticket_id")) != ticket_id:
            continue
        parsed = _parse_datetime(action.get("created_at"))
        if parsed is not None:
            relevant.append(parsed)
    if not relevant:
        return None
    first_action = min(relevant)
    delta = first_action - created_at
    return max(0, int(delta.total_seconds() // 60))


def _compute_avg_first_response_minutes(tickets: list[dict[str, Any]]) -> int:
    actions = list_ticket_actions(limit=200)
    values = [
        minutes
        for ticket in tickets
        if (minutes := _first_response_minutes_for_ticket(ticket, actions)) is not None
    ]
    if not values:
        return 0
    return int(round(sum(values) / len(values)))


def get_dashboard_metrics() -> dict[str, int | float]:
    """Aggregate headline metrics from the active ticket source."""
    tickets = get_active_tickets()
    total = len(tickets)
    open_count = sum(
        1 for item in tickets if str(item.get("status", "")).lower() in ACTIVE_STATUSES
    )
    threshold = get_review_threshold()
    needs_review = sum(
        1 for item in tickets if float(item.get("confidence", threshold)) < threshold
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
        "avg_first_response_minutes": _compute_avg_first_response_minutes(tickets),
    }


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


def get_draft_for_ticket(ticket_id: str) -> str:
    """Return the latest saved draft for the ticket or a generated fallback."""
    latest = get_latest_draft(str(ticket_id))
    if latest and str(latest.get("text", "")).strip():
        return str(latest["text"])
    ticket = get_ticket_by_id(ticket_id)
    if ticket is not None:
        queue_name = str(ticket.get("queue", "general")).lower()
        subject_name = str(ticket.get("subject", "this support request"))
        return (
            f"Thanks for reporting the {queue_name} issue about {subject_name}. "
            "I’ve reviewed the latest context and will confirm the next step with "
            "the right team before sending the final update."
        )
    return "Thanks for your patience while we review this issue."


def get_ticket_sources(ticket: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the knowledge-base passages supporting the current draft."""
    latest = get_latest_draft(str(ticket.get("ticket_id", "")))
    if latest and latest.get("sources"):
        return latest["sources"]
    queue = str(ticket.get("queue", "general")).lower()
    return [
        {
            "id": f"KBASE-{queue.upper()}-1",
            "score": 0.9,
            "text": (
                f"Support guidance for {queue} tickets: validate the account state, "
                "confirm recent changes, and route to the correct team before "
                "sending a response."
            ),
        },
        {
            "id": f"KBASE-{queue.upper()}-2",
            "score": 0.8,
            "text": (
                "Review the full ticket context, recent policies, and any "
                "customer-impacting incidents before closing the case."
            ),
        },
    ]


def _safe_action_wrapper(action_name: str, operation: Any) -> ActionResult:
    try:
        result = operation()
        if isinstance(result, ActionResult):
            return result
        return ActionResult(True, result if isinstance(result, str) else action_name, result)
    except Exception as exc:  # pragma: no cover
        logger.exception("Action failed: %s", action_name)
        return ActionResult(False, f"{action_name} failed: {exc}", {})


def save_draft_in_repo(
    ticket_id: str,
    *,
    text: str,
    source: str = "agent",
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Repository helper to persist a new draft version."""
    return repo_save_draft(
        str(ticket_id),
        text=str(text),
        source=str(source),
        sources=sources or [],
        grounded=1,
    )


def save_draft(
    ticket_id: str,
    *,
    text: str,
    source: str = "agent",
    sources: list[dict[str, Any]] | None = None,
) -> ActionResult:
    """Persist a new draft version from the UI or reviewer."""

    def _operation() -> ActionResult:
        cleaned = (text or "").strip()
        if not cleaned:
            return ActionResult(False, "Draft cannot be empty.", {"ticket_id": ticket_id})
        draft = save_draft_in_repo(ticket_id, text=cleaned, source=source, sources=sources or [])
        update_ticket_fields(ticket_id, status="in_review")
        log_ticket_decision(ticket_id, action="edited", agent="agent")
        return ActionResult(
            True,
            f"Draft saved as version {draft['version']}.",
            {"draft": draft},
        )

    return _safe_action_wrapper("Save draft", _operation)


def regenerate_draft(ticket_id: str, *, agent: str = "agent") -> ActionResult:
    """Regenerate a new draft using RAG context or a template fallback."""

    def _operation() -> ActionResult:
        ticket = get_ticket_by_id(str(ticket_id)) or {
            "subject": "support request",
            "queue": "general",
        }
        sources: list[dict[str, Any]] = []
        generated = ""
        source = "template_fallback"
        try:
            from copilot.rag import retriever as rag_retriever

            passages = rag_retriever.retrieve(ticket, limit=3)
            sources = [
                {"id": f"FAQ-{index + 1}", "score": 0.9 - index * 0.05, "text": passage}
                for index, passage in enumerate(passages)
            ]
            if sources:
                try:
                    from copilot.rag import generator as rag_generator

                    candidate = rag_generator.generate_answer(ticket, passages=passages)
                    if isinstance(candidate, str) and candidate.strip():
                        generated = candidate.strip()
                        source = "rag"
                except Exception:
                    logger.info("RAG generator unavailable; using template fallback.")
        except Exception:
            logger.info("RAG retriever unavailable; using template fallback.")

        if not generated:
            queue_name = str(ticket.get("queue", "general")).lower()
            subject_name = str(ticket.get("subject", "this support request"))
            problem_text = str(ticket.get("body") or "Customer reported this issue.")
            generated = (
                f"Thanks for reporting the {queue_name} issue about {subject_name}. "
                f"The current ticket states: {problem_text} "
                "I’ve reviewed the current context and am confirming the next step "
                "with the right team before sending the final update."
            )
            source = "template_fallback"
            if not sources:
                sources = [
                    {
                        "id": f"TEMPLATE-{queue_name.upper()}-1",
                        "score": 0.75,
                        "text": problem_text,
                    }
                ]

        draft = save_draft_in_repo(ticket_id, text=generated, source=source, sources=sources)
        update_ticket_fields(ticket_id, status="in_review")
        log_ticket_decision(ticket_id, action="edited", agent=agent)
        return ActionResult(
            True,
            f"Draft regenerated to version {draft['version']} ({source}).",
            {"draft": draft, "source": source},
        )

    return _safe_action_wrapper("Regenerate draft", _operation)


def send_reply(ticket_id: str, *, text: str, agent: str = "agent") -> ActionResult:
    """Persist the submitted reply without mutating the original ticket body."""

    def _operation() -> ActionResult:
        cleaned = (text or "").strip()
        if not cleaned:
            return ActionResult(False, "Reply text is empty.", {"ticket_id": ticket_id})
        draft = get_latest_draft(str(ticket_id))
        original = str(draft.get("text", "")) if draft else ""
        outcome = "accepted" if cleaned == original else "edited"
        distance = 0
        if outcome != "accepted":
            distance = _edit_distance(cleaned, original) if original else len(cleaned)
        add_outcome(
            ticket_id=str(ticket_id),
            draft_id=draft.get("id") if draft else None,
            outcome=outcome,
            edit_distance=int(distance),
            final_text=cleaned,
        )
        update_ticket_fields(str(ticket_id), status="resolved")
        decision = "accepted" if outcome == "accepted" else "edited"
        log_ticket_decision(str(ticket_id), action=decision, agent=agent)
        return ActionResult(
            True,
            f"Reply {outcome} and ticket marked resolved.",
            {"ticket_id": ticket_id, "outcome": outcome, "edit_distance": int(distance)},
        )

    return _safe_action_wrapper("Send reply", _operation)


def reject_draft(ticket_id: str, *, text: str | None = None, agent: str = "agent") -> ActionResult:
    """Reject the current draft and send it back to the queue."""
    def _operation() -> ActionResult:
        payload = text or get_draft_for_ticket(str(ticket_id))
        draft = get_latest_draft(str(ticket_id))
        add_outcome(
            ticket_id=str(ticket_id),
            draft_id=draft.get("id") if draft else None,
            outcome="rejected",
            edit_distance=0,
            final_text=payload,
        )
        update_ticket_fields(str(ticket_id), status="new")
        log_ticket_decision(str(ticket_id), action="rejected", agent=agent)
        return ActionResult(
            True,
            "Draft rejected and returned to the queue.",
            {"ticket_id": ticket_id},
        )

    return _safe_action_wrapper("Reject draft", _operation)


def escalate(ticket_id: str, *, agent: str = "agent") -> ActionResult:
    """Escalate the ticket and assign the team from the routing config."""
    def _operation() -> ActionResult:
        ticket = get_ticket_by_id(str(ticket_id)) or {}
        queue = str(ticket.get("queue", "general")).lower()
        team = get_queue_team_map().get(queue, "Support Desk")
        update_ticket_fields(str(ticket_id), status="escalated", team=team)
        log_ticket_decision(str(ticket_id), action="escalated", agent=agent)
        return ActionResult(
            True,
            "Ticket escalated to the incident workflow.",
            {"ticket_id": ticket_id, "team": team},
        )

    return _safe_action_wrapper("Escalate ticket", _operation)


def confirm_routing(ticket_id: str, *, agent: str = "agent") -> ActionResult:
    """Confirm the current routing suggestions and mark the ticket as assigned."""
    def _operation() -> ActionResult:
        update_ticket_fields(str(ticket_id), status="assigned")
        log_ticket_decision(str(ticket_id), action="confirmed", agent=agent)
        return ActionResult(True, "Routing confirmed.", {"ticket_id": ticket_id})

    return _safe_action_wrapper("Confirm routing", _operation)


def correct_routing(
    ticket_id: str,
    *,
    queue: str,
    priority: str,
    team: str | None = None,
    agent: str = "agent",
) -> ActionResult:
    """Record an explicit routing correction and update the ticket record."""
    def _operation() -> ActionResult:
        ticket = get_ticket_by_id(str(ticket_id)) or {}
        predicted_queue = str(ticket.get("queue", "general")).lower()
        predicted_priority = str(ticket.get("priority", "medium")).lower()
        chosen_queue = str(queue).lower()
        chosen_priority = str(priority).lower()
        team_name = team or get_queue_team_map().get(chosen_queue, "Support Desk")
        add_correction(
            ticket_id=str(ticket_id),
            predicted_queue=predicted_queue,
            final_queue=chosen_queue,
            predicted_priority=predicted_priority,
            final_priority=chosen_priority,
            team=team_name,
            agent=agent,
        )
        update_ticket_fields(
            str(ticket_id),
            queue=chosen_queue,
            priority=chosen_priority,
            team=team_name,
            status="in_review",
        )
        log_ticket_decision(str(ticket_id), action="edited", agent=agent)
        return ActionResult(
            True,
            f"Routing corrected to {chosen_queue} / {chosen_priority}.",
            {
                "ticket_id": ticket_id,
                "queue": chosen_queue,
                "priority": chosen_priority,
                "team": team_name,
            },
        )

    return _safe_action_wrapper("Correct routing", _operation)


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
    threshold = get_review_threshold()
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
        if needs_review_only and float(ticket.get("confidence", threshold)) >= threshold:
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


def format_ticket_age(ticket: dict[str, Any]) -> str:
    """Return a compact human-readable age for a ticket."""
    raw = ticket.get("created_at")
    if not raw:
        return "recently"
    created = _parse_datetime(raw)
    if created is None:
        return "recently"
    minutes = max(0, int((datetime.now(timezone.utc) - created).total_seconds() // 60))
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


def set_ticket_status_for_ui(ticket_id: str, *, status: str) -> ActionResult:
    """Convenience wrapper for UI status updates."""
    def _operation() -> ActionResult:
        set_ticket_status(str(ticket_id), str(status))
        return ActionResult(
            True,
            f"Ticket status set to {status}.",
            {"ticket_id": ticket_id, "status": status},
        )

    return _safe_action_wrapper("Update ticket status", _operation)


def list_drafts_for_ui(ticket_id: str) -> list[dict[str, Any]]:
    """Read the persisted drafts for a ticket."""
    return list_drafts(str(ticket_id))
