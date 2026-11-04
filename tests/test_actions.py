"""Every agent action must change the database, never only show a toast."""
from __future__ import annotations

from app.services import ticket_service as svc
from copilot.db import repository as repo


def _rows(table: str, ticket_id: str) -> list[dict]:
    with repo._connect() as conn:
        cur = conn.execute(f"SELECT * FROM {table} WHERE ticket_id = ?", (ticket_id,))
        return [dict(r) for r in cur.fetchall()]


def _ticket(ticket_id: str) -> dict:
    return next(t for t in repo.list_tickets() if t["ticket_id"] == ticket_id)


def test_send_reply_resolves_and_keeps_customer_message(make_ticket):
    original = make_ticket("T-SEND")["body"]
    result = svc.send_reply("T-SEND", text="Hello, we fixed the timeout. Please retry.")
    assert result.ok, result.message
    ticket = _ticket("T-SEND")
    assert ticket["status"] == "resolved"
    assert ticket["body"] == original, "send_reply must NEVER overwrite the customer's message"
    outcomes = _rows("outcomes", "T-SEND")
    assert len(outcomes) == 1
    assert outcomes[0]["final_text"].startswith("Hello, we fixed")


def test_send_reply_unchanged_draft_is_accepted_and_edited_is_edited(make_ticket):
    make_ticket("T-ACC")
    svc.save_draft("T-ACC", text="Draft A", source="ai")
    svc.send_reply("T-ACC", text="Draft A")
    make_ticket("T-EDIT")
    svc.save_draft("T-EDIT", text="Draft A", source="ai")
    svc.send_reply("T-EDIT", text="Draft A, edited by the agent")
    assert _rows("outcomes", "T-ACC")[0]["outcome"] == "accepted"
    edited = _rows("outcomes", "T-EDIT")[0]
    assert edited["outcome"] == "edited" and edited["edit_distance"] > 0


def test_send_reply_rejects_empty_text(make_ticket):
    make_ticket("T-EMPTY")
    result = svc.send_reply("T-EMPTY", text="   ")
    assert not result.ok
    assert _ticket("T-EMPTY")["status"] != "resolved"


def test_save_draft_creates_increasing_versions(make_ticket):
    make_ticket("T-VER")
    svc.save_draft("T-VER", text="v1")
    svc.save_draft("T-VER", text="v2")
    versions = [d["version"] for d in repo.list_drafts("T-VER")]
    assert sorted(versions) == [1, 2]


def test_regenerate_adds_a_new_version_with_sources(make_ticket):
    make_ticket("T-REGEN")
    first = svc.regenerate_draft("T-REGEN")
    second = svc.regenerate_draft("T-REGEN")
    assert first.ok and second.ok
    assert len(repo.list_drafts("T-REGEN")) == 2
    assert repo.get_latest_draft("T-REGEN")["sources"], "draft must carry its sources"


def test_reject_logs_outcome_and_returns_ticket_to_queue(make_ticket):
    make_ticket("T-REJ", status="in_review")
    svc.save_draft("T-REJ", text="Bad draft")
    result = svc.reject_draft("T-REJ")
    assert result.ok
    assert _rows("outcomes", "T-REJ")[0]["outcome"] == "rejected"
    assert _ticket("T-REJ")["status"] == "new"


def test_escalate_sets_status_and_team(make_ticket):
    make_ticket("T-ESC")
    assert svc.escalate("T-ESC").ok
    ticket = _ticket("T-ESC")
    assert ticket["status"] == "escalated"
    assert ticket.get("team")


def test_confirm_routing_marks_assigned(make_ticket):
    make_ticket("T-CONF")
    assert svc.confirm_routing("T-CONF").ok
    assert _ticket("T-CONF")["status"] == "assigned"


def test_correct_routing_saves_correction_for_retraining(make_ticket):
    make_ticket("T-FIX", queue="technical_support", priority="high")
    result = svc.correct_routing("T-FIX", queue="billing_and_payments", priority="low")
    assert result.ok
    correction = _rows("corrections", "T-FIX")[0]
    assert correction["predicted_queue"] == "technical_support"
    assert correction["final_queue"] == "billing_and_payments"
    ticket = _ticket("T-FIX")
    assert ticket["queue"] == "billing_and_payments" and ticket["priority"] == "low"


def test_actions_never_raise_into_the_ui():
    result = svc.send_reply("DOES-NOT-EXIST", text="hi")
    assert hasattr(result, "ok") and hasattr(result, "message")
