"""Validate that generated answers are grounded in retrieved support context."""

from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "has", "have", "if", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "we", "will", "with", "you",
    "your", "our", "can", "not", "do", "does", "please", "thanks", "thank",
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def grounding_check(
    answer: str | None = None,
    sources: list[dict[str, Any]] | list[str] | None = None,
    *,
    min_coverage: float = 0.25,
) -> dict[str, Any]:
    """Assess answer grounding before human review."""
    if not answer or not sources:
        return {
            "grounded": False,
            "coverage": 0.0,
            "unsupported_sentences": [],
            "reason": "missing_answer_or_sources",
        }
    source_text = " ".join(
        source.get("text", "") if isinstance(source, dict) else str(source)
        for source in sources
    )
    source_terms = _terms(source_text)
    answer_terms = _terms(answer)
    if not answer_terms:
        return {
            "grounded": False,
            "coverage": 0.0,
            "unsupported_sentences": [],
            "reason": "empty_answer_terms",
        }
    covered = answer_terms & source_terms
    coverage = len(covered) / len(answer_terms)
    unsupported: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        sentence_terms = _terms(sentence)
        if sentence_terms and len(sentence_terms & source_terms) / len(sentence_terms) < 0.1:
            unsupported.append(sentence.strip())
    return {
        "grounded": coverage >= min_coverage,
        "coverage": round(coverage, 4),
        "unsupported_sentences": unsupported[:5],
        "reason": "ok" if coverage >= min_coverage else "low_source_overlap",
    }
