"""Retrain from persisted human corrections without discarding the real training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from copilot.db import repository as repo
from copilot.models.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _feedback_frame(task: str) -> pd.DataFrame:
    corrections = repo.list_corrections()
    if not corrections:
        return pd.DataFrame(columns=["subject", "body", task])
    tickets = {str(ticket["ticket_id"]): ticket for ticket in repo.list_tickets()}
    rows: list[dict[str, str]] = []
    for correction in corrections:
        ticket = tickets.get(str(correction["ticket_id"]))
        if ticket is None:
            continue
        label = correction.get("final_queue") if task == "queue" else correction.get("final_priority")
        if not label:
            continue
        rows.append(
            {
                "subject": str(ticket.get("subject", "")),
                "body": str(ticket.get("body", "")),
                "queue" if task == "queue" else "priority": str(label),
            }
        )
    return pd.DataFrame(rows)


def retrain_from_feedback(
    task: str = "queue",
    *,
    output_dir: str | Path | None = None,
    use_distilbert: bool = False,
) -> dict[str, Any]:
    """Append real human corrections to the training split and retrain."""
    feedback = _feedback_frame(task)
    if feedback.empty:
        return {"task": task, "status": "no_feedback", "n_feedback": 0}
    train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split missing: {train_path}")
    base = pd.read_csv(train_path)
    combined = pd.concat([base, feedback], ignore_index=True)
    processed = PROJECT_ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    feedback_path = processed / f"retrain_{task}.csv"
    combined.to_csv(feedback_path, index=False)
    result = train(
        data_path=feedback_path,
        output_dir=output_dir or PROJECT_ROOT / "models" / f"{task}_feedback_model",
        task=task,
        use_distilbert=use_distilbert,
    )
    gate = float(
        (yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8")) or {}).get(
            "f1_gate", 0.70
        )
    )
    result["n_feedback"] = int(len(feedback))
    result["training_rows"] = int(len(combined))
    result["f1_gate"] = gate
    result["promotion_eligible"] = bool(float(result.get("macro_f1", 0.0)) >= gate)
    result["status"] = "retrained" if result["promotion_eligible"] else "below_gate"
    (PROJECT_ROOT / "reports" / f"retrain_{task}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["queue", "priority"], required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            retrain_from_feedback(args.task, output_dir=args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
