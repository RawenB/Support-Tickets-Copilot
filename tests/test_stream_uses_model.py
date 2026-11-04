"""The simulated stream must feed TEXT to the model, never the dataset's true labels."""
from __future__ import annotations

import sqlite3

from copilot.db import repository as repo
from copilot.incidents import simulate_stream as sim
from copilot.models import predict as predict_module


def test_stream_stores_model_predictions_not_ground_truth(tmp_db, monkeypatch):
    monkeypatch.setattr(
        sim,
        "_english_records",
        lambda: [
            {
                "ticket_id": "S-1",
                "subject": "Double charge",
                "body": "I was charged twice.",
                "queue": "billing_and_payments",  # ground truth from the dataset
                "priority": "low",
                "customer_email": "a@example.com",
                "language": "en",
            }
        ],
    )
    monkeypatch.setattr(sim.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        predict_module,
        "predict",
        lambda text, task="queue", **_: {
            "task": task,
            "label": "technical_support" if task == "queue" else "critical",
            "confidence": 0.55,
        },
    )
    sim.simulate_stream(rate_seconds=0, limit=1)
    ticket = next(t for t in repo.list_tickets() if t["ticket_id"] == "S-1")
    assert ticket["queue"] == "technical_support", "stream copied the true label, not the prediction"
    assert ticket["priority"] == "critical"
    assert abs(ticket["confidence"] - 0.55) < 1e-6, "confidence must come from the model"
    with sqlite3.connect(tmp_db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tickets)")}
    assert "true_queue" in cols, "keep ground truth in a separate column for evaluation only"


def test_stream_rate_zero_does_not_fall_back_to_config_sleep(tmp_db, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(sim, "_english_records", lambda: [
        {"ticket_id": "S-2", "subject": "s", "body": "b", "queue": "q", "priority": "low",
         "customer_email": "x@example.com", "language": "en"}])
    monkeypatch.setattr(sim.time, "sleep", lambda s: slept.append(s))
    sim.simulate_stream(rate_seconds=0, limit=1)
    assert slept == [0.0], f"rate_seconds=0 was ignored, slept {slept}"
