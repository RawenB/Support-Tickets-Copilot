"""Regenerate must show a NEW draft in the editor and keep the same ticket open."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from copilot.db import repository as repo

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _button(at: AppTest, label: str):
    return next(b for b in at.button if label in b.label)


@pytest.fixture()
def app(make_ticket):
    make_ticket("R-1", priority="critical", subject="API timeout", body="API calls time out for many users.")
    make_ticket("R-2", priority="low", subject="Invoice question", body="Why was I charged twice?")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def _editor(at: AppTest, ticket_id: str = "R-1") -> str:
    return at.text_area(key=f"draft_{ticket_id}").value


def test_first_open_creates_a_real_draft_not_a_hardcoded_sentence(app):
    text = _editor(app)
    assert text.strip()
    assert "identified the likely fix" not in text
    assert [d["version"] for d in repo.list_drafts("R-1")] == [1]


def test_regenerate_keeps_the_same_ticket_selected(app):
    _button(app, "Regenerate").click().run()
    assert not app.exception, app.exception
    assert app.session_state["selected_ticket_id"] == "R-1"


def test_regenerate_puts_the_new_draft_into_the_editor(app):
    first = _editor(app)
    _button(app, "Regenerate").click().run()
    second = _editor(app)
    latest = repo.get_latest_draft("R-1")
    assert latest["version"] == 2
    assert second == latest["text"], "editor must show the newly generated draft"
    assert second != first, "regenerate must change the text"


def test_each_regeneration_gives_a_different_draft(app):
    seen = [_editor(app)]
    for _ in range(2):
        _button(app, "Regenerate").click().run()
        seen.append(_editor(app))
    assert len(set(seen)) == 3


def test_save_edit_keeps_ticket_open_and_stores_agent_version(app):
    app.text_area(key="draft_R-1").set_value("My own edited reply").run()
    _button(app, "Save edit").click().run()
    assert app.session_state["selected_ticket_id"] == "R-1"
    latest = repo.get_latest_draft("R-1")
    assert latest["text"] == "My own edited reply" and latest["source"] == "agent"


def test_send_reply_still_moves_to_the_next_ticket(app):
    _button(app, "Send reply").click().run()
    assert app.session_state["selected_ticket_id"] == "R-2"
