"""Click the real Streamlit buttons and check the database."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from copilot.db import repository as repo

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _outcomes(ticket_id: str) -> list[dict]:
    with repo._connect() as conn:
        rows = conn.execute("SELECT * FROM outcomes WHERE ticket_id = ?", (ticket_id,))
        return [dict(r) for r in rows.fetchall()]


def _button(at: AppTest, label: str):
    return next(b for b in at.button if label in b.label)


@pytest.fixture()
def app_with_ticket(make_ticket):
    make_ticket("UI-1", priority="critical", status="new")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_send_reply_button_resolves_ticket(app_with_ticket):
    at = app_with_ticket
    _button(at, "Send reply").click().run()
    ticket = next(t for t in repo.list_tickets() if t["ticket_id"] == "UI-1")
    assert ticket["status"] == "resolved"
    assert len(_outcomes("UI-1")) == 1


def test_reject_button_logs_rejected_outcome(app_with_ticket):
    _button(app_with_ticket, "Reject").click().run()
    assert [o["outcome"] for o in _outcomes("UI-1")] == ["rejected"]


def test_escalate_button_escalates(app_with_ticket):
    _button(app_with_ticket, "Escalate").click().run()
    ticket = next(t for t in repo.list_tickets() if t["ticket_id"] == "UI-1")
    assert ticket["status"] == "escalated"


def test_regenerate_button_adds_draft_version(app_with_ticket):
    _button(app_with_ticket, "Regenerate").click().run()
    assert len(repo.list_drafts("UI-1")) >= 1
