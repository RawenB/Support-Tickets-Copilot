"""Repository helpers for reading and writing support data from SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "support_copilot.db"


def _connect() -> sqlite3.Connection:
    """Open a SQLite connection for the support-copilot database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=30.0,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    connection.execute("PRAGMA busy_timeout = 30000;")
    return connection


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_ticket(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    record["ticket_id"] = str(record.get("ticket_id") or record.get("id") or "TKT-0000")
    record.setdefault(
        "customer_name",
        record.get("customer_name") or record.get("customer") or "Unknown customer",
    )
    record.setdefault(
        "customer_email",
        record.get("customer_email") or record.get("email") or "unknown@example.com",
    )
    record.setdefault("subject", record.get("subject") or "Untitled ticket")
    record.setdefault("body", record.get("body") or "")
    record.setdefault("queue", record.get("queue") or "general")
    record.setdefault("priority", record.get("priority") or "medium")
    record.setdefault("true_queue", record.get("true_queue") or record.get("queue") or "general")
    record.setdefault(
        "true_priority",
        record.get("true_priority") or record.get("priority") or "medium",
    )
    record.setdefault("status", record.get("status") or "new")
    record.setdefault("confidence", float(record.get("confidence") or 0.75))
    record.setdefault("team", record.get("team") or "Support Desk")
    record.setdefault("created_at", (_utc_now() - timedelta(minutes=12)).isoformat())
    record.setdefault("assigned_agent", record.get("assigned_agent") or "Unassigned")
    return record


def _ensure_ticket_columns(connection: sqlite3.Connection) -> None:
    """Add newer columns to an existing tickets table when the schema is older."""
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(tickets)").fetchall()
    }
    if "team" not in columns:
        connection.execute(
            "ALTER TABLE tickets ADD COLUMN team TEXT NOT NULL DEFAULT 'Support Desk'"
        )
    if "true_queue" not in columns:
        connection.execute("ALTER TABLE tickets ADD COLUMN true_queue TEXT")
    if "true_priority" not in columns:
        connection.execute("ALTER TABLE tickets ADD COLUMN true_priority TEXT")
    if "spike_id" not in columns:
        connection.execute("ALTER TABLE tickets ADD COLUMN spike_id TEXT")


def _ensure_draft_columns(connection: sqlite3.Connection) -> None:
    """Add generation metadata columns to an older drafts table."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(drafts)").fetchall()}
    for name, definition in (
        ("backend", "TEXT"),
        ("model_name", "TEXT"),
        ("prompt_hash", "TEXT"),
        ("latency_ms", "REAL"),
    ):
        if name not in columns:
            connection.execute(f"ALTER TABLE drafts ADD COLUMN {name} {definition}")


def initialize_database() -> None:
    """Create the SQLite schema used by the product UI and admin events."""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                queue TEXT NOT NULL,
                priority TEXT NOT NULL,
                true_queue TEXT,
                true_priority TEXT,
                spike_id TEXT,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                customer_name TEXT NOT NULL,
                customer_email TEXT NOT NULL,
                assigned_agent TEXT NOT NULL,
                team TEXT NOT NULL DEFAULT 'Support Desk',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_ticket_columns(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                action TEXT NOT NULL,
                agent TEXT NOT NULL,
                details TEXT,
                decision TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                queue TEXT NOT NULL,
                priority TEXT NOT NULL,
                customer_email TEXT NOT NULL,
                language TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'ai',
                sources_json TEXT,
                grounded INTEGER NOT NULL DEFAULT 1,
                backend TEXT,
                model_name TEXT,
                prompt_hash TEXT,
                latency_ms REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_draft_columns(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                predicted_queue TEXT,
                final_queue TEXT,
                predicted_priority TEXT,
                final_priority TEXT,
                team TEXT,
                agent TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                draft_id INTEGER,
                outcome TEXT NOT NULL,
                edit_distance INTEGER NOT NULL DEFAULT 0,
                final_text TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # Rows are seeded only by explicit pipeline/stream scripts. Database
        # initialization stays idempotent and never invents product data.


def _review_threshold() -> float:
    """Return the configured review threshold from params.yaml."""
    params_path = PROJECT_ROOT / "params.yaml"
    if not params_path.exists():
        return 0.70
    with params_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    threshold = payload.get("review_threshold", 0.70)
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return 0.70


def list_tickets(*, limit: int | None = None, needs_review: bool = False) -> list[dict[str, Any]]:
    """Fetch tickets from SQLite sorted by priority and recency."""
    initialize_database()
    threshold = _review_threshold()
    with _connect() as connection:
        query = "SELECT * FROM tickets"
        if needs_review:
            query += f" WHERE confidence < {threshold}"
        query += (
            " ORDER BY CASE priority "
            "WHEN 'critical' THEN 0 "
            "WHEN 'high' THEN 1 "
            "WHEN 'medium' THEN 2 "
            "WHEN 'low' THEN 3 "
            "WHEN 'very_low' THEN 4 "
            "ELSE 5 END, created_at DESC"
        )
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = connection.execute(query).fetchall()
    return [_normalize_ticket(dict(row)) for row in rows]


def upsert_ticket(record: dict[str, Any]) -> dict[str, Any]:
    """Insert or update a ticket record, returning the canonical record."""
    ticket = _normalize_ticket(record)
    now = _utc_now().isoformat()
    ticket["updated_at"] = now
    if not ticket.get("created_at"):
        ticket["created_at"] = now
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO tickets (
                ticket_id, subject, body, queue, priority, true_queue, true_priority,
                spike_id, status, confidence, customer_name, customer_email, assigned_agent,
                team, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                subject = excluded.subject,
                body = excluded.body,
                queue = excluded.queue,
                priority = excluded.priority,
                true_queue = excluded.true_queue,
                true_priority = excluded.true_priority,
                spike_id = excluded.spike_id,
                status = excluded.status,
                confidence = excluded.confidence,
                customer_name = excluded.customer_name,
                customer_email = excluded.customer_email,
                assigned_agent = excluded.assigned_agent,
                team = excluded.team,
                updated_at = excluded.updated_at
            """,
            (
                ticket["ticket_id"],
                ticket["subject"],
                ticket["body"],
                ticket["queue"],
                ticket["priority"],
                ticket.get("true_queue", ticket["queue"]),
                ticket.get("true_priority", ticket["priority"]),
                ticket.get("spike_id"),
                ticket["status"],
                ticket["confidence"],
                ticket["customer_name"],
                ticket["customer_email"],
                ticket["assigned_agent"],
                ticket.get("team", "Support Desk"),
                ticket["created_at"],
                ticket["updated_at"],
            ),
        )
    return ticket


def set_ticket_status(ticket_id: str, status: str) -> dict[str, Any]:
    """Persist a new ticket lifecycle state."""
    with _connect() as connection:
        connection.execute(
            "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
            (status, _utc_now().isoformat(), ticket_id),
        )
    return {"ticket_id": ticket_id, "status": status}


def update_ticket_fields(ticket_id: str, **fields: Any) -> dict[str, Any]:
    """Update selected ticket fields in one transaction."""
    data = dict(fields)
    if not data:
        return {"ticket_id": ticket_id}
    assignments = ["updated_at = ?"]
    values: list[Any] = [_utc_now().isoformat()]
    for key, value in data.items():
        if value is None:
            continue
        assignments.append(f"{key} = ?")
        values.append(value)
    values.append(ticket_id)
    with _connect() as connection:
        connection.execute(
            f"UPDATE tickets SET {', '.join(assignments)} WHERE ticket_id = ?",
            tuple(values),
        )
    return {"ticket_id": ticket_id, **data}


def save_draft(
    ticket_id: str,
    *,
    text: str,
    source: str = "ai",
    sources: list[dict[str, Any]] | None = None,
    grounded: int = 1,
    backend: str | None = None,
    model_name: str | None = None,
    prompt_hash: str | None = None,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    """Persist a draft version for the ticket and auto-increment the version."""
    with _connect() as connection:
        current_version = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM drafts WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()[0]
        version = int(current_version) + 1
        created_at = _utc_now().isoformat()
        cursor = connection.execute(
            """
            INSERT INTO drafts (
                ticket_id, version, text, source, sources_json, grounded,
                backend, model_name, prompt_hash, latency_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                version,
                text,
                source,
                json.dumps(sources or [], ensure_ascii=False),
                int(bool(grounded)),
                backend,
                model_name,
                prompt_hash,
                latency_ms,
                created_at,
            ),
        )
    return {
        "id": cursor.lastrowid,
        "ticket_id": ticket_id,
        "version": version,
        "text": text,
        "source": source,
        "sources": sources or [],
        "grounded": int(bool(grounded)),
        "backend": backend,
        "model_name": model_name,
        "prompt_hash": prompt_hash,
        "latency_ms": latency_ms,
        "created_at": created_at,
    }


def get_latest_draft(ticket_id: str) -> dict[str, Any] | None:
    """Fetch the latest draft for a ticket."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM drafts WHERE ticket_id = ? ORDER BY version DESC, id DESC LIMIT 1",
            (ticket_id,),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["sources"] = json.loads(data.get("sources_json") or "[]")
    return data


def list_drafts(ticket_id: str) -> list[dict[str, Any]]:
    """Return all draft versions for a ticket."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM drafts WHERE ticket_id = ? ORDER BY version DESC, id DESC",
            (ticket_id,),
        ).fetchall()
    drafts: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(item.get("sources_json") or "[]")
        drafts.append(item)
    return drafts


def add_correction(
    *,
    ticket_id: str,
    predicted_queue: str,
    final_queue: str,
    predicted_priority: str,
    final_priority: str,
    team: str,
    agent: str = "agent",
) -> dict[str, Any]:
    """Record a routing correction made by the human reviewer."""
    created_at = _utc_now().isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO corrections (
                ticket_id, predicted_queue, final_queue, predicted_priority,
                final_priority, team, agent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                predicted_queue,
                final_queue,
                predicted_priority,
                final_priority,
                team,
                agent,
                created_at,
            ),
        )
    return {
        "id": cursor.lastrowid,
        "ticket_id": ticket_id,
        "predicted_queue": predicted_queue,
        "final_queue": final_queue,
        "predicted_priority": predicted_priority,
        "final_priority": final_priority,
        "team": team,
        "agent": agent,
        "created_at": created_at,
    }


def list_corrections(*, limit: int | None = None) -> list[dict[str, Any]]:
    """Return human routing corrections newest-first."""
    initialize_database()
    with _connect() as connection:
        query = "SELECT * FROM corrections ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = connection.execute(query).fetchall()
    return [dict(row) for row in rows]


def add_outcome(
    *,
    ticket_id: str,
    draft_id: int | None,
    outcome: str,
    edit_distance: int = 0,
    final_text: str | None = None,
) -> dict[str, Any]:
    """Record a final disposition for a draft and its edited field values."""
    created_at = _utc_now().isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO outcomes (
                ticket_id, draft_id, outcome, edit_distance, final_text, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, draft_id, outcome, int(edit_distance), final_text, created_at),
        )
    return {
        "id": cursor.lastrowid,
        "ticket_id": ticket_id,
        "draft_id": draft_id,
        "outcome": outcome,
        "edit_distance": int(edit_distance),
        "final_text": final_text,
        "created_at": created_at,
    }


def list_outcomes(ticket_id: str | None = None) -> list[dict[str, Any]]:
    """Return persisted ticket outcomes in newest-first order."""
    with _connect() as connection:
        if ticket_id is None:
            rows = connection.execute(
                "SELECT * FROM outcomes ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM outcomes WHERE ticket_id = ? ORDER BY created_at DESC",
                (ticket_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def log_ticket_action(
    ticket_id: str,
    *,
    action: str,
    agent: str = "system",
    details: str | None = None,
    decision: str | None = None,
) -> dict[str, Any]:
    """Persist a human action taken against a ticket."""
    now = _utc_now().isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO ticket_actions (ticket_id, action, agent, details, decision, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, action, agent, details, decision, now),
        )
    return {
        "id": cursor.lastrowid,
        "ticket_id": ticket_id,
        "action": action,
        "agent": agent,
        "details": details,
        "decision": decision,
        "created_at": now,
    }


def list_ticket_actions(*, limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent human actions recorded against tickets."""
    initialize_database()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM ticket_actions ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_dashboard_metrics() -> dict[str, int | float]:
    """Return the queue metrics used by the top bar and main inbox."""
    tickets = list_tickets()
    threshold = _review_threshold()
    open_count = sum(1 for item in tickets if item["status"] in {"new", "in_review", "assigned"})
    needs_review_count = sum(1 for item in tickets if item["confidence"] < threshold)
    escalated_count = sum(1 for item in tickets if item["status"] == "escalated")
    avg_response = 8
    return {
        "open": open_count,
        "needs_review": needs_review_count,
        "escalated": escalated_count,
        "avg_first_response_minutes": avg_response,
    }


def get_repository() -> dict[str, Any]:
    """Return the repository layer used by the Streamlit and review flows."""
    initialize_database()
    return {"database_path": str(DB_PATH), "tickets": list_tickets()}


def reset_database() -> None:
    """Utility for tests and admin resets; removes the SQLite file."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    initialize_database()
