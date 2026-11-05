"""Simulate ticket activity streams for incident watch experimentation."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from copilot.db.repository import DB_PATH, initialize_database, upsert_ticket
from copilot.models import predict as predict_module

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UNCLASSIFIED_QUEUE = "unclassified"
LOWEST_PRIORITY = "very_low"

logger = logging.getLogger("support_copilot.simulate_stream")


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "incident.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_params() -> dict[str, Any]:
    params_path = PROJECT_ROOT / "params.yaml"
    if not params_path.exists():
        return {}
    with params_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _default_replay_rate() -> float:
    params = _load_params()
    value = params.get("replay_rate_seconds")
    if value is None:
        value = params.get("incident", {}).get("replay_rate_seconds")
    if value is None:
        value = _load_config().get("incident", {}).get("replay_rate_seconds", 3)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 3.0


def _predict_queue(text: str) -> tuple[str, float]:
    try:
        prediction = predict_module.predict(text, task="queue")
        return str(prediction["label"]), float(prediction["confidence"])
    except FileNotFoundError:
        logger.warning(
            "Queue model artifact is missing; storing ticket as '%s' with zero confidence.",
            UNCLASSIFIED_QUEUE,
        )
        return UNCLASSIFIED_QUEUE, 0.0
    except Exception:
        logger.exception("Queue prediction failed; storing ticket as unclassified.")
        return UNCLASSIFIED_QUEUE, 0.0


def _predict_priority(text: str) -> tuple[str, float]:
    try:
        prediction = predict_module.predict(text, task="priority")
        return str(prediction["label"]), float(prediction["confidence"])
    except FileNotFoundError:
        logger.warning(
            "Priority model artifact is missing; storing ticket as '%s' with zero confidence.",
            LOWEST_PRIORITY,
        )
        return LOWEST_PRIORITY, 0.0
    except Exception:
        logger.exception("Priority prediction failed; storing lowest priority.")
        return LOWEST_PRIORITY, 0.0


def _predict_ticket(text: str) -> tuple[str, str, float]:
    queue_label, queue_confidence = _predict_queue(text)
    priority_label, priority_confidence = _predict_priority(text)
    return queue_label, priority_label, min(queue_confidence, priority_confidence)


def _english_records() -> list[dict[str, Any]]:
    split_path = PROJECT_ROOT / "data" / "splits" / "test.csv"
    if split_path.exists():
        frame = pd.read_csv(split_path)
    else:
        csv_path = PROJECT_ROOT / "data" / "raw" / "customer-support-tickets.csv"
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
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
    true_queue: str | None = None,
    true_priority: str | None = None,
    spike_id: str | None = None,
    created_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Insert a simulated event into SQLite and return the stored ticket payload."""
    initialize_database()
    if isinstance(created_at, datetime):
        created_value = created_at.isoformat()
    else:
        created_value = str(created_at or datetime.now(timezone.utc).isoformat())
    ticket = {
        "ticket_id": str(ticket_id),
        "subject": subject,
        "body": body,
        "queue": queue,
        "priority": priority,
        "true_queue": true_queue or queue,
        "true_priority": true_priority or priority,
        "spike_id": spike_id,
        "status": "new",
        "confidence": confidence,
        "customer_name": customer_name or "Unknown customer",
        "customer_email": customer_email,
        "assigned_agent": "Unassigned",
        "created_at": created_value,
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
    """Replay real tickets from one predicted queue as an incident burst."""
    start = start_time or datetime.now(timezone.utc)
    records = _english_records()
    if not records:
        logger.warning("No real tickets are available to build a spike cluster.")
        return []

    target_queue = str(queue).lower()
    prepared: list[tuple[dict[str, Any], str, str, float]] = []
    sample_size = min(len(records), max(200, int(count) * 50))
    for row in records[:sample_size]:
        text = f"{row.get('subject', '')} {row.get('body', '')}".strip()
        predicted_queue, predicted_priority, confidence = _predict_ticket(text)
        prepared.append((row, predicted_queue, predicted_priority, confidence))
        if (
            str(predicted_queue).lower() == target_queue
            and sum(1 for item in prepared if str(item[1]).lower() == target_queue) >= count
        ):
            break

    candidates = [
        item for item in prepared if str(item[1]).lower() == target_queue
    ]
    if not candidates:
        candidates = [
            (row, target_queue, priority, confidence)
            for row, _predicted_queue, priority, confidence in prepared
        ]

    spike_id = f"SPIKE-{start.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    cluster: list[dict[str, Any]] = []
    for offset in range(count):
        row, predicted_queue, predicted_priority, confidence = candidates[
            offset % len(candidates)
        ]
        drift = timedelta(minutes=offset * 5 / max(count - 1, 1))
        created = start + drift
        cluster.append(
            insert_ticket_event(
                ticket_id=f"{spike_id}-{offset:04d}",
                subject=str(row.get("subject") or "Support request"),
                body=str(row.get("body") or ""),
                queue=predicted_queue,
                priority=predicted_priority,
                customer_email=str(
                    row.get("customer_email") or f"spike{offset + 1}@example.com"
                ),
                language=str(row.get("language") or "en"),
                customer_name=f"Spike customer {offset + 1}",
                confidence=confidence,
                true_queue=str(row.get("queue") or "general"),
                true_priority=str(row.get("priority") or "medium"),
                spike_id=spike_id,
                created_at=created,
            )
        )
    return cluster


def _write_stream_eval(inserted: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    labelled = [row for row in inserted if row.get("true_queue") and row.get("true_priority")]
    if not labelled:
        return {"n": 0}
    true_queue = [str(row["true_queue"]) for row in labelled]
    pred_queue = [str(row["queue"]) for row in labelled]
    true_priority = [str(row["true_priority"]) for row in labelled]
    pred_priority = [str(row["priority"]) for row in labelled]
    labels = sorted(set(true_queue) | set(pred_queue))
    payload = {
        "n": int(len(labelled)),
        "queue": {
            "macro_f1": float(f1_score(true_queue, pred_queue, average="macro")),
            "accuracy": float(accuracy_score(true_queue, pred_queue)),
            "confusion_matrix": confusion_matrix(true_queue, pred_queue, labels=labels).tolist(),
            "labels": labels,
        },
        "priority": {
            "macro_f1": float(f1_score(true_priority, pred_priority, average="macro")),
            "accuracy": float(accuracy_score(true_priority, pred_priority)),
            "confusion_matrix": confusion_matrix(
                true_priority, pred_priority, labels=sorted(set(true_priority) | set(pred_priority))
            ).tolist(),
            "labels": sorted(set(true_priority) | set(pred_priority)),
        },
    }
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stream_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_incident_stream(inserted: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist the replay as the DVC incident stream and summarize its shape."""
    stream_path = PROJECT_ROOT / "data" / "interim" / "incident_stream.csv"
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "ticket_id",
        "subject",
        "queue",
        "priority",
        "true_queue",
        "true_priority",
        "spike_id",
        "confidence",
        "customer_email",
        "created_at",
    ]
    frame = pd.DataFrame(inserted)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame[columns].to_csv(stream_path, index=False)
    metrics = {
        "n_events": int(len(frame)),
        "n_spikes": int(frame["spike_id"].dropna().nunique()) if "spike_id" in frame else 0,
        "queue_counts": frame["queue"].value_counts().to_dict() if "queue" in frame else {},
        "priority_counts": frame["priority"].value_counts().to_dict()
        if "priority" in frame
        else {},
    }
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "incident_stream_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def simulate_stream(
    *, rate_seconds: float | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Replay recorded English tickets from the test split into the SQLite ticket table."""
    rate = _default_replay_rate() if rate_seconds is None else float(rate_seconds)
    rows = _english_records()
    if not rows:
        return []
    selected = rows[:limit] if limit else rows
    inserted: list[dict[str, Any]] = []
    for row in selected:
        text = f"{row['subject']} {row['body']}"
        queue_label, priority_label, confidence = _predict_ticket(text)
        inserted.append(
            insert_ticket_event(
                ticket_id=row["ticket_id"],
                subject=row["subject"],
                body=row["body"],
                queue=queue_label,
                priority=priority_label,
                customer_email=row["customer_email"],
                language=row["language"],
                confidence=confidence,
                true_queue=str(row.get("queue") or "general"),
                true_priority=str(row.get("priority") or "medium"),
            )
        )
        time.sleep(rate)
    if limit is None:
        _write_stream_eval(inserted)
        _write_incident_stream(inserted)
    return inserted


if __name__ == "__main__":
    simulate_stream()
