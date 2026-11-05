"""Generate a grounded draft answer from retrieved support context."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from typing import Any

STYLES = ("concise", "detailed", "friendly")


def _first_name(ticket: dict[str, Any]) -> str:
    name = str(ticket.get("customer_name") or "").strip()
    if not name or name.lower() in {"unknown customer", "test customer"}:
        return "there"
    return name.split()[0].title()


def _sentences(text: str, limit: int) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return parts[:limit]


def _source_list(
    passages: list[str] | None, sources: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    if sources:
        return sources
    return [
        {"id": f"FAQ-{index + 1:02d}", "text": passage}
        for index, passage in enumerate(passages or [])
    ]


def _evidence(sources: list[dict[str, Any]], variant: int) -> tuple[str, list[str]]:
    if not sources:
        return "", []
    ordered = sources[variant % len(sources) :] + sources[: variant % len(sources)]
    selected = ordered[:2]
    citations = [str(source.get("id", "")) for source in selected if source.get("id")]
    text = " ".join(
        " ".join(_sentences(str(source.get("text", "")), 2)) for source in selected
    )
    return text.strip(), citations


def _citations(citations: list[str]) -> str:
    return " ".join(f"[{citation}]" for citation in citations if citation)


def _extractive(ticket: dict[str, Any], sources: list[dict[str, Any]], variant: int) -> str:
    subject = str(ticket.get("subject") or "your request").strip()
    queue = str(ticket.get("queue") or "support").replace("_", " ").strip()
    name = _first_name(ticket)
    guidance, citations = _evidence(sources, variant)
    cited = _citations(citations)
    style = STYLES[variant % len(STYLES)]
    if style == "concise":
        body = guidance or "We are checking the account and the relevant logs now."
        return (
            f"Hi {name}, thanks for contacting us about \"{subject}\". {body} {cited} "
            "We will update you as soon as the next step is confirmed.\n\n"
            "Best regards,\nSupport Team"
        )
    if style == "detailed":
        lines = [
            f"Hello {name},",
            "",
            f"Thank you for reaching out about \"{subject}\" ({queue}).",
            "",
            "Based on the support guidance:",
            f"1. {guidance or 'We are reviewing the account state and recent activity.'} {cited}",
            "2. If the issue continues, reply with the exact error text and the time it occurred.",
            "",
            "We will keep this ticket open until the next action is confirmed.",
            "",
            "Kind regards,",
            "Support Team",
        ]
        return "\n".join(lines)
    body = guidance or "I am checking this with the right team now."
    return (
        f"Hi {name}, sorry for the trouble with \"{subject}\" - we know how frustrating that is. "
        f"{body} {cited} Please reply here if anything changes, and I will stay on it until "
        "the next step is confirmed.\n\nWarm regards,\nYour Support Team"
    )


def _prompt(ticket: dict[str, Any], sources: list[dict[str, Any]], variant: int) -> str:
    context = "\n".join(
        f"[{source.get('id')}] {source.get('text')}" for source in sources
    )
    return (
        f"Write a support reply for this ticket using only the FAQ context. "
        f"Cite sources like [FAQ-01]. Variant {variant}.\n"
        f"Subject: {ticket.get('subject')}\nBody: {ticket.get('body')}\n"
        f"Context:\n{context}"
    )


def _ollama(prompt: str) -> str | None:
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(
        os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode()).get("response")
    except Exception:
        return None


def _hf_api(prompt: str) -> str | None:
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        return None
    model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    request = urllib.request.Request(
        f"https://api-inference.huggingface.co/models/{model}",
        data=json.dumps({"inputs": prompt}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
            if isinstance(payload, list) and payload:
                return str(payload[0].get("generated_text", ""))
    except Exception:
        return None
    return None


def generate_answer_with_metadata(
    ticket: dict[str, Any] | str | None = None,
    *,
    passages: list[str] | None = None,
    sources: list[dict[str, Any]] | None = None,
    variant: int = 0,
    backend: str | None = None,
) -> dict[str, Any]:
    """Generate a draft and return the backend, model, prompt hash, sources, and latency."""
    started = time.perf_counter()
    if isinstance(ticket, dict):
        resolved_sources = _source_list(passages, sources)
        prompt = _prompt(ticket, resolved_sources, variant)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        chosen = (backend or os.environ.get("RAG_BACKEND", "extractive")).lower()
        text = None
        model_name = "extractive"
        if chosen == "ollama":
            model_name = os.environ.get("OLLAMA_MODEL", "llama3.2")
            text = _ollama(prompt)
        elif chosen == "hf_api":
            model_name = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
            text = _hf_api(prompt)
        if not text:
            chosen = "extractive"
            model_name = "extractive"
            text = _extractive(ticket, resolved_sources, variant)
        if resolved_sources and not any(f"[{source.get('id')}]" in text for source in resolved_sources):
            text = f"{text} {_citations([str(source.get('id')) for source in resolved_sources[:2]])}"
        return {
            "text": text,
            "backend": chosen,
            "model_name": model_name,
            "prompt_hash": prompt_hash,
            "sources": resolved_sources,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    text = (
        "Thanks for reporting this issue. I have reviewed the latest context and am "
        "confirming the next step with the right team before sending the final update."
    )
    return {
        "text": text,
        "backend": "extractive",
        "model_name": "extractive",
        "prompt_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        "sources": [],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def generate_answer(
    ticket: dict[str, Any] | str | None = None,
    *,
    passages: list[str] | None = None,
    sources: list[dict[str, Any]] | None = None,
    variant: int = 0,
    backend: str | None = None,
) -> str:
    """Generate a grounded draft response; `variant` changes style and evidence order."""
    return generate_answer_with_metadata(
        ticket,
        passages=passages,
        sources=sources,
        variant=variant,
        backend=backend,
    )["text"]
