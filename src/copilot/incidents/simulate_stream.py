"""Simulate ticket activity streams for incident watch experimentation."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from copilot.db.repository import DB_PATH, initialize_database, upsert_ticket

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "incident.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _english_records() -> list[dict[str, Any]]:
    parquet_path = PROJECT_ROOT / "data" / "splits" / "test.parquet"
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    else:
        csv_path = PROJECT_ROOT / "data" / "raw" / "customer-support-tickets.csv"
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
        else:
            demo_path = PROJECT_ROOT / "data" / "demo_tickets.json"
            if demo_path.exists():
                frame = pd.read_json(demo_path)
            else:
                return []
    if "language" in frame.columns:
        frame = frame[frame["language"].fillna("").str.lower() == "en"].copy()
    elif "text" in frame.columns:
        frame = frame[frame["text"].fillna("").str.len() > 0].copy()
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        subject = str(row.get("subject") or row.get("title") or f"Ticket {index}")
        body = str(row.get("body") or row.get("message") or row.get("text") or "Support request")
        queue = str(row.get("queue") or "general")
        priority = str(row.get("priority") or "medium")
        email = str(row.get("customer_email") or row.get("email") or f"user{index}@example.com")
        language = str(row.get("language") or "en")
        records.append(
            {
                "ticket_id": str(row.get("ticket_id") or row.get("id") or f"SIM-{index:04d}"),
                "subject": subject,
                "body": body,
                "queue": queue,
                "priority": priority,
                "customer_email": email,
                "language": language,
            }
        )
    return records


def insert_ticket_event(
    *,
    ticket_id: str,
    subject: str,
    body: str,
    queue: str,
    priority: str,
    customer_email: str,
    language: str = "en",
    customer_name: str | None = None,
    confidence: float = 0.82,
) -> dict[str, Any]:
    """Insert a simulated event into SQLite and return the stored ticket payload."""
    initialize_database()
    ticket = {
        "ticket_id": str(ticket_id),
        "subject": subject,
        "body": body,
        "queue": queue,
        "priority": priority,
        "status": "new",
        "confidence": confidence,
        "customer_name": customer_name or "Demo customer",
        "customer_email": customer_email,
        "assigned_agent": "Unassigned",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    stored = upsert_ticket(ticket)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO ticket_events (
                ticket_id, subject, queue, priority, customer_email, language, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stored["ticket_id"],
                stored["subject"],
                stored["queue"],
                stored["priority"],
                stored["customer_email"],
                language,
                stored["created_at"],
            ),
        )
    return stored


def create_spike_cluster(
    *,
    start_time: datetime | None = None,
    count: int = 30,
    queue: str = "technical",
) -> list[dict[str, Any]]:
    """Generate a 30-ticket spike within a 5-minute window.

    This is used by the admin controls to simulate an incident burst.
    """
    start = start_time or datetime.now(timezone.utc)
    cluster: list[dict[str, Any]] = []
    for offset in range(count):
        drift = timedelta(minutes=offset * 5 / max(count - 1, 1))
        created = start + drift
        payload = {
            "ticket_id": f"SPIKE-{offset:04d}",
            "subject": f"Service degradation alert {offset + 1}",
            "body": "Multiple customers are reporting the same issue at nearly the same time.",
            "queue": queue,
            "priority": "critical" if offset < 3 else "high",
            "status": "new",
            "confidence": 0.74,
            "customer_name": f"Spike customer {offset + 1}",
            "customer_email": f"spike{offset + 1}@example.com",
            "assigned_agent": "Unassigned",
            "created_at": created.isoformat(),
        }
        cluster.append(upsert_ticket(payload))
    return cluster


def simulate_stream(
    *, rate_seconds: float | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Replay recorded English tickets from the test split into the SQLite ticket table."""
    config = _load_config()
    rate = float(rate_seconds or config.get("incident", {}).get("replay_rate_seconds", 3))
    rows = _english_records()
    if not rows:
        return []
    selected = rows[:limit] if limit else rows
    inserted: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        inserted.append(
            insert_ticket_event(
                ticket_id=row["ticket_id"],
                subject=row["subject"],
                body=row["body"],
                queue=row["queue"],
                priority=row["priority"],
                customer_email=row["customer_email"],
                language=row["language"],
                confidence=0.82 if index % 3 else 0.76,
            )
        )
        time.sleep(rate)
    return inserted


if __name__ == "__main__":
    simulate_stream()
