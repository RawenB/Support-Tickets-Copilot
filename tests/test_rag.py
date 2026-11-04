from __future__ import annotations

from copilot.rag.faq_index import build_faq_index
from copilot.rag.generator import generate_answer
from copilot.rag.grounding_check import grounding_check
from copilot.rag.retriever import retrieve


def test_rag_modules_importable() -> None:
    assert callable(build_faq_index)
    assert callable(retrieve)
    assert callable(generate_answer)
    assert callable(grounding_check)
