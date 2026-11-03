"""Repository helpers for reading and writing support data from SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "support_copilot.db"


def _connect() -> sqlite3.Connection:
    """Open a SQLite connection for the support-copilot database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_ticket(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    record["ticket_id"] = str(record.get("ticket_id") or record.get("id") or "TKT-0000")
    record.setdefault(
        "customer_name", record.get("customer_name") or record.get("customer") or "Unknown customer"
    )
    record.setdefault(
        "customer_email",
        record.get("customer_email") or record.get("email") or "unknown@example.com",
    )
    record.setdefault("subject", record.get("subject") or "Untitled ticket")
    record.setdefault("body", record.get("body") or "")
    record.setdefault("queue", record.get("queue") or "general")
    record.setdefault("priority", record.get("priority") or "medium")
    record.setdefault("status", record.get("status") or "new")
    record.setdefault("confidence", float(record.get("confidence") or 0.75))
    record.setdefault("created_at", (_utc_now() - timedelta(minutes=12)).isoformat())
    record.setdefault("assigned_agent", record.get("assigned_agent") or "Unassigned")
    return record


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
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                customer_name TEXT NOT NULL,
                customer_email TEXT NOT NULL,
                assigned_agent TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
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

        if connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0:
            demo_path = PROJECT_ROOT / "data" / "demo_tickets.json"
            if demo_path.exists():
                with demo_path.open("r", encoding="utf-8") as handle:
                    rows = json.load(handle)
                for row in rows:
                    ticket = _normalize_ticket(
                        {
                            "ticket_id": row.get("id", "TKT-1"),
                            "subject": row.get("subject"),
                            "body": row.get("body"),
                            "queue": row.get("queue"),
                            "priority": row.get("priority"),
                            "status": row.get("status"),
                            "confidence": row.get("confidence", 0.86),
                            "customer_name": row.get("customer"),
                            "customer_email": row.get("email"),
                            "assigned_agent": row.get("agent"),
                            "created_at": (
                                datetime.now(timezone.utc) - timedelta(minutes=12)
                            ).isoformat(),
                        }
                    )
                    connection.execute(
                        """
                        INSERT INTO tickets (
                            ticket_id, subject, body, queue, priority, status, confidence,
                            customer_name, customer_email, assigned_agent, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ticket["ticket_id"],
                            ticket["subject"],
                            ticket["body"],
                            ticket["queue"],
                            ticket["priority"],
                            ticket["status"],
                            ticket["confidence"],
                            ticket["customer_name"],
                            ticket["customer_email"],
                            ticket["assigned_agent"],
                            ticket["created_at"],
                            ticket["created_at"],
                        ),
                    )


def list_tickets(*, limit: int | None = None, needs_review: bool = False) -> list[dict[str, Any]]:
    """Fetch tickets from SQLite sorted by priority and recency."""
    initialize_database()
    with _connect() as connection:
        query = "SELECT * FROM tickets"
        if needs_review:
            query += " WHERE confidence < 0.80"
        query += " ORDER BY priority DESC, created_at DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = connection.execute(query).fetchall()
    return [_normalize_ticket(dict(row)) for row in rows]


def upsert_ticket(record: dict[str, Any]) -> dict[str, Any]:
    """Insert or update a ticket record, returning the canonical record."""
    ticket = _normalize_ticket(record)
    now = _utc_now().isoformat()
    ticket["updated_at"] = now
    if "created_at" not in ticket or not ticket["created_at"]:
        ticket["created_at"] = now
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO tickets (
                ticket_id, subject, body, queue, priority, status, confidence,
                customer_name, customer_email, assigned_agent, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                subject = excluded.subject,
                body = excluded.body,
                queue = excluded.queue,
                priority = excluded.priority,
                status = excluded.status,
                confidence = excluded.confidence,
                customer_name = excluded.customer_name,
                customer_email = excluded.customer_email,
                assigned_agent = excluded.assigned_agent,
                updated_at = excluded.updated_at
            """,
            (
                ticket["ticket_id"],
                ticket["subject"],
                ticket["body"],
                ticket["queue"],
                ticket["priority"],
                ticket["status"],
                ticket["confidence"],
                ticket["customer_name"],
                ticket["customer_email"],
                ticket["assigned_agent"],
                ticket["created_at"],
                ticket["updated_at"],
            ),
        )
    return ticket


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
    open_count = sum(1 for item in tickets if item["status"] in {"new", "in_review", "assigned"})
    needs_review_count = sum(1 for item in tickets if item["confidence"] < 0.8)
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
