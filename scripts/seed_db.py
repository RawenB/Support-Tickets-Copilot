from __future__ import annotations

import json
from pathlib import Path

from copilot.db.repository import initialize_database, upsert_ticket

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    initialize_database()
    demo_path = PROJECT_ROOT / "data" / "demo_tickets.json"
    if not demo_path.exists():
        return
    with demo_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for item in payload:
        upsert_ticket(
            {
                "ticket_id": str(item.get("id") or item.get("ticket_id") or "TKT-000"),
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
                "assigned_agent": str(
                    item.get("agent") or item.get("assigned_agent") or "Unassigned"
                ),
                "team": str(item.get("team") or "Support Desk"),
                "created_at": str(item.get("created_at") or "2026-11-04T12:00:00+00:00"),
            }
        )


if __name__ == "__main__":
    main()
