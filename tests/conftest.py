from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate = str(path)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Isolated SQLite DB so tests never touch data/support_copilot.db."""
    from copilot.db import repository as repo

    db_file = tmp_path / "test_support_copilot.db"
    monkeypatch.setattr(repo, "DB_PATH", db_file)
    # modules that did `from ... import DB_PATH` keep their own reference
    try:
        from copilot.incidents import simulate_stream as sim

        monkeypatch.setattr(sim, "DB_PATH", db_file)
    except Exception:  # pragma: no cover
        pass
    repo.initialize_database()
    return db_file


@pytest.fixture()
def make_ticket(tmp_db):
    from copilot.db.repository import upsert_ticket

    def _make(ticket_id="T-1", **overrides):
        record = {
            "ticket_id": ticket_id,
            "subject": "API calls are timing out",
            "body": "Several users cannot access the workspace since 20 minutes.",
            "queue": "technical_support",
            "priority": "high",
            "status": "new",
            "confidence": 0.9,
            "customer_name": "Test Customer",
            "customer_email": "test@example.com",
            "assigned_agent": "Unassigned",
            "created_at": "2026-11-04T10:00:00+00:00",
        }
        record.update(overrides)
        return upsert_ticket(record)

    return _make
