"""Evaluate retrieval and grounded generation on real support tickets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from copilot.rag.generator import generate_answer_with_metadata
from copilot.rag.grounding_check import grounding_check
from copilot.rag.retriever import retrieve_sources
from copilot.utils.mlflow_utils import setup_mlflow
from copilot.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "rag_eval.json"


def _ticket(row: pd.Series) -> dict[str, Any]:
    return {
        "ticket_id": str(row.get("ticket_id", "")),
        "subject": str(row.get("subject", "")),
        "body": str(row.get("body", "")),
        "queue": str(row.get("queue", "")),
        "priority": str(row.get("priority", "")),
        "customer_name": str(row.get("customer_name", "")),
    }


def evaluate_rag(*, limit: int = 40, seed: int = 42) -> dict[str, Any]:
    """Run retrieval and generation on a real test sample and persist metrics."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing {TEST_PATH}; run the data pipeline first.")
    frame = pd.read_csv(TEST_PATH)
    if frame.empty:
        raise ValueError("Test split is empty.")
    sample = frame.sample(n=min(limit, len(frame)), random_state=seed)
    top1_matches = 0
    top3_matches = 0
    grounded = 0
    latencies: list[float] = []
    backends: Counter[str] = Counter()
    answer_lengths: list[int] = []
    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        ticket = _ticket(row)
        sources = retrieve_sources(ticket, limit=3)
        expected_queue = ticket["queue"].lower()
        if sources and str(sources[0].get("id", "")) and expected_queue:
            document_queue = ""
            from copilot.rag.retriever import _load_meta

            documents = {doc["id"]: doc for doc in _load_meta().get("documents", [])}
            document_queue = str(documents.get(sources[0]["id"], {}).get("queue", "")).lower()
            top1_matches += int(document_queue == expected_queue)
            top3_matches += int(
                any(
                    str(documents.get(source["id"], {}).get("queue", "")).lower()
                    == expected_queue
                    for source in sources
                )
            )
        result = generate_answer_with_metadata(ticket, sources=sources, variant=0)
        check = grounding_check(result["text"], sources)
        grounded += int(bool(check["grounded"]))
        latencies.append(float(result["latency_ms"]))
        backends[str(result["backend"])] += 1
        answer_lengths.append(len(result["text"]))
        rows.append(
            {
                "ticket_id": ticket["ticket_id"],
                "expected_queue": expected_queue,
                "source_ids": [source["id"] for source in sources],
                "grounded": bool(check["grounded"]),
                "coverage": check["coverage"],
                "backend": result["backend"],
                "latency_ms": result["latency_ms"],
            }
        )
    total = len(sample)
    metrics = {
        "n_tickets": total,
        "retrieval_top1_queue_match": round(top1_matches / total, 4),
        "retrieval_top3_queue_match": round(top3_matches / total, 4),
        "grounded_rate": round(grounded / total, 4),
        "avg_latency_ms": round(sum(latencies) / total, 2),
        "avg_answer_chars": round(sum(answer_lengths) / total, 2),
        "backend_counts": dict(backends),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_file": str(TEST_PATH.relative_to(PROJECT_ROOT)),
        "split": "test",
        "sample_limit": limit,
        "seed": seed,
        "metrics": metrics,
        "rows": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    setup_mlflow("support-copilot-rag")
    with mlflow.start_run(run_name="rag-eval"):
        mlflow.log_params({"limit": limit, "seed": seed, "split": "test"})
        mlflow.log_metrics(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.log_artifact(str(REPORT_PATH))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG assistant on real tickets.")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    print(json.dumps(evaluate_rag(limit=args.limit, seed=args.seed)["metrics"], indent=2))


if __name__ == "__main__":
    main()
