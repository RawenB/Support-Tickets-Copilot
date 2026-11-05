"""Build and maintain the support FAQ index used for retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAQ_DIR = PROJECT_ROOT / "data" / "faq"
INDEX_PATH = FAQ_DIR / "index.faiss"
META_PATH = FAQ_DIR / "index_meta.json"
EMBEDDER = "all-MiniLM-L6-v2"
FAQ_PATTERN = re.compile(r"^#\s*(FAQ-\d+)\s*:\s*(.+)$")


def load_faq_documents() -> list[dict[str, Any]]:
    """Read curated FAQ markdown files into structured documents."""
    documents: list[dict[str, Any]] = []
    for path in sorted(FAQ_DIR.glob("faq-*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        match = FAQ_PATTERN.match(lines[0].strip())
        if not match:
            continue
        faq_id, title = match.group(1), match.group(2).strip()
        body_lines = [
            line for line in lines[1:] if line.strip() and not line.lower().startswith("queue:")
        ]
        queue = ""
        for line in lines[1:]:
            if line.lower().startswith("queue:"):
                queue = line.split(":", 1)[1].strip()
                break
        text = " ".join(" ".join(body_lines).split())
        documents.append(
            {
                "id": faq_id,
                "title": title,
                "queue": queue,
                "text": text,
                "source": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return documents


def build_faq_index() -> dict[str, Any]:
    """Embed all FAQ documents and persist a FAISS index plus metadata."""
    documents = load_faq_documents()
    if not documents:
        raise FileNotFoundError(f"No FAQ markdown files found in {FAQ_DIR}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDER)
    embeddings = model.encode(
        [f"{doc['title']}. {doc['text']}" for doc in documents],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    try:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, str(INDEX_PATH))
        backend = "faiss"
    except Exception:
        import numpy as np

        np.save(FAQ_DIR / "index.npy", embeddings.astype("float32"))
        backend = "numpy"
    META_PATH.write_text(
        json.dumps(
            {
                "embedder": EMBEDDER,
                "backend": backend,
                "documents": documents,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "n_documents": len(documents),
        "embedder": EMBEDDER,
        "backend": backend,
        "index_path": str(INDEX_PATH),
    }


def main() -> None:
    print(json.dumps(build_faq_index(), indent=2))


if __name__ == "__main__":
    main()
