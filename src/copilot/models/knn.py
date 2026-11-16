"""Similarity-weighted k-nearest-neighbour ticket classifier ("ticket memory").

Support tickets recur: about half of the test tickets are paraphrases of an earlier ticket,
and paraphrases share the label 95 %+ of the time. Looking up the most similar past
tickets and voting with sharpened similarity weights (``similarity ** power``) beats both
linear TF-IDF models and fine-tuned transformers on this data. Three views of similarity
make different mistakes, so they are blended (see ``copilot.models.blend``):

- ``char``: character 3-5-gram TF-IDF (robust to typos and inflections)
- ``word``: word 1-2-gram TF-IDF
- ``embedding``: sentence-transformer embeddings (paraphrases with different wording)
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer

REPRESENTATIONS = ("char", "word", "embedding")


class KNNTicketModel(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        representation: str = "char",
        *,
        k: int = 15,
        power: float = 10.0,
        embedder: str = "all-MiniLM-L6-v2",
        batch_size: int = 512,
    ):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        self.representation = representation
        self.k = int(k)
        self.power = float(power)
        self.embedder = embedder
        self.batch_size = int(batch_size)
        self._encoder: Any = None

    # -- representation -------------------------------------------------------------
    def _make_vectorizer(self) -> TfidfVectorizer | None:
        if self.representation == "char":
            return TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2)
        if self.representation == "word":
            return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
        return None

    def _sentence_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            from copilot.models.transformer import select_device

            self._encoder = SentenceTransformer(self.embedder, device=str(select_device()))
        return self._encoder

    def _encode(self, texts: Sequence[str]) -> Any:
        if self.representation == "embedding":
            return self._sentence_encoder().encode(
                list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
        return self.vectorizer_.transform(texts)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_encoder"] = None
        return state

    # -- sklearn-style API ---------------------------------------------------------
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "KNNTicketModel":
        texts = [str(t) for t in texts]
        self.classes_ = np.asarray(sorted({str(label) for label in labels}))
        index = {label: i for i, label in enumerate(self.classes_)}
        self.label_ids_ = np.asarray([index[str(label)] for label in labels], dtype=np.int64)
        self.vectorizer_ = self._make_vectorizer()
        if self.vectorizer_ is not None:
            self.vectorizer_.fit(texts)
        self.memory_ = self._encode(texts)
        return self

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        query = self._encode([str(t) for t in texts])
        n_query = query.shape[0]
        n_classes = len(self.classes_)
        k = max(1, min(self.k, self.memory_.shape[0]))
        output = np.zeros((n_query, n_classes))
        for start in range(0, n_query, self.batch_size):
            block = query[start : start + self.batch_size] @ self.memory_.T
            sims = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
            if k < sims.shape[1]:
                top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            else:
                top = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
            weights = np.take_along_axis(sims, top, axis=1).clip(min=0) ** self.power
            neighbour_labels = self.label_ids_[top]
            for c in range(n_classes):
                output[start : start + sims.shape[0], c] = (weights * (neighbour_labels == c)).sum(axis=1)
        totals = output.sum(axis=1, keepdims=True)
        empty = totals[:, 0] <= 1e-12
        output[empty] = 1.0 / n_classes  # no similar ticket at all: uninformative vote
        totals[empty] = 1.0
        return output / totals

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]
