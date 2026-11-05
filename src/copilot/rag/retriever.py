"""Retrieve top support documents for a ticket via vector search."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAQ_DIR = PROJECT_ROOT / "data" / "faq"
META_PATH = FAQ_DIR / "index_meta.json"
INDEX_PATH = FAQ_DIR / "index.faiss"
NUMPY_INDEX_PATH = FAQ_DIR / "index.npy"


def _query_text(ticket: dict[str, Any] | str | None) -> str:
    if isinstance(ticket, str):
        return ticket
    if isinstance(ticket, dict):
        return " ".join(
            str(ticket.get(key, ""))
            for key in ("subject", "body", "queue", "priority")
        )
    return "support ticket"


@lru_cache(maxsize=1)
def _load_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        from copilot.rag.faq_index import build_faq_index

        build_faq_index()
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_embedder() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_load_meta().get("embedder", "all-MiniLM-L6-v2"))


@lru_cache(maxsize=1)
def _load_faiss() -> Any | None:
    try:
        import faiss

        return faiss.read_index(str(INDEX_PATH))
    except Exception:
        return None


def _keyword_sources(query: str, documents: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    tokens = {token for token in query.lower().split() if len(token) > 2}
    scored: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        haystack = f"{document['title']} {document['text']} {document.get('queue', '')}".lower()
        overlap = sum(1 for token in tokens if token in haystack)
        score = overlap / max(1, len(tokens))
        scored.append((score, document))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": document["id"],
            "title": document["title"],
            "text": document["text"],
            "score": round(float(score), 4),
            "source": document["source"],
        }
        for score, document in scored[:limit]
    ]


def retrieve_sources(
    ticket: dict[str, Any] | str | None = None,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return ranked FAQ sources with real IDs and scores."""
    meta = _load_meta()
    documents = meta.get("documents", [])
    if not documents:
        return []
    query = _query_text(ticket)
    faiss_index = _load_faiss()
    expected_queue = (
        str(ticket.get("queue", "")).lower() if isinstance(ticket, dict) else ""
    )
    if faiss_index is not None:
        embedder = _load_embedder()
        vector = embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        candidate_count = min(max(limit * 4, 10), len(documents))
        scores, indices = faiss_index.search(vector, candidate_count)
        candidates: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):  # noqa: B905 (Python 3.9)
            if int(index) < 0:
                continue
            document = documents[int(index)]
            boost = 0.12 if expected_queue and document.get("queue") == expected_queue else 0.0
            candidates.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "text": document["text"],
                    "score": round(float(score) + boost, 4),
                    "source": document["source"],
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]
    if NUMPY_INDEX_PATH.exists():
        embeddings = np.load(NUMPY_INDEX_PATH)
        embedder = _load_embedder()
        vector = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = embeddings @ np.asarray(vector)
        order = np.argsort(scores)[::-1][:limit]
        return [
            {
                "id": documents[int(index)]["id"],
                "title": documents[int(index)]["title"],
                "text": documents[int(index)]["text"],
                "score": round(float(scores[int(index)]), 4),
                "source": documents[int(index)]["source"],
            }
            for index in order
        ]
    return _keyword_sources(query, documents, limit)


def retrieve(
    ticket: dict[str, Any] | str | None = None,
    *,
    limit: int = 3,
) -> list[str]:
    """Return the most relevant FAQ context for a ticket or issue description."""
    return [source["text"] for source in retrieve_sources(ticket, limit=limit)]
