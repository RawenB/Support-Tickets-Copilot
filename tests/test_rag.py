from __future__ import annotations

from copilot.rag.faq_index import build_faq_index, load_faq_documents
from copilot.rag.generator import generate_answer, generate_answer_with_metadata
from copilot.rag.grounding_check import grounding_check
from copilot.rag.retriever import retrieve, retrieve_sources


def test_rag_modules_importable() -> None:
    assert callable(build_faq_index)
    assert callable(retrieve)
    assert callable(generate_answer)
    assert callable(grounding_check)


def test_faq_corpus_has_25_real_documents() -> None:
    documents = load_faq_documents()
    assert len(documents) == 25
    assert len({document["id"] for document in documents}) == 25
    assert all(document["text"].strip() for document in documents)


def test_retrieval_returns_real_faq_ids_and_citations() -> None:
    ticket = {
        "subject": "Duplicate charge",
        "body": "I was charged twice for the same invoice.",
        "queue": "billing_and_payments",
    }
    sources = retrieve_sources(ticket, limit=3)
    assert sources
    assert all(source["id"].startswith("FAQ-") for source in sources)
    answer = generate_answer(ticket, sources=sources)
    assert f"[{sources[0]['id']}]" in answer


def test_generation_records_backend_metadata_and_grounding() -> None:
    ticket = {
        "subject": "Password reset loop",
        "body": "The reset link keeps returning to the login page.",
        "queue": "technical_support",
    }
    sources = retrieve_sources(ticket, limit=3)
    first = generate_answer_with_metadata(ticket, sources=sources, variant=0)
    second = generate_answer_with_metadata(ticket, sources=sources, variant=1)
    assert first["text"] != second["text"]
    assert first["backend"] == "extractive"
    assert first["prompt_hash"] != second["prompt_hash"]
    assert first["sources"]
    assert grounding_check(first["text"], sources)["grounded"]
