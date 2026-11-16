"""Probability blending of complementary ticket classifiers.

The dataset contains many paraphrased near-duplicates of earlier tickets. A high-capacity
TF-IDF model is very good at recognising those, while a fine-tuned transformer generalises
to genuinely new tickets. A weighted average of their probabilities gets the best of both;
the weight is chosen on the validation split only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score


def aligned_proba(model: Any, texts: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Return ``model.predict_proba`` with columns re-ordered to ``labels``."""
    proba = np.asarray(model.predict_proba(list(texts)), dtype=np.float64)
    classes = [str(label) for label in model.classes_]
    output = np.zeros((proba.shape[0], len(labels)))
    for column, label in enumerate(labels):
        if label in classes:
            output[:, column] = proba[:, classes.index(label)]
    return output


def tune_weights(
    probas: dict[str, np.ndarray],
    y_true: Sequence[str],
    labels: Sequence[str],
    step: float = 0.1,
) -> tuple[dict[str, float], float]:
    """Exhaustive search over the weight simplex (weights are multiples of ``step``, sum 1)."""
    names = list(probas)
    units = int(round(1 / step))
    label_array = np.asarray(labels)
    y_true = list(y_true)

    def compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for first in range(total, -1, -1):
            for rest in compositions(total - first, parts - 1):
                yield (first, *rest)

    best: tuple[dict[str, float], float] = ({names[0]: 1.0}, -1.0)
    for combo in compositions(units, len(names)):
        mix = sum(c * probas[n] for c, n in zip(combo, names) if c)
        score = float(f1_score(y_true, label_array[mix.argmax(axis=1)], average="macro"))
        if score > best[1] + 1e-9:
            best = ({n: c / units for c, n in zip(combo, names)}, score)
    return best


def tune_weight(
    probas: Sequence[np.ndarray],
    y_true: Sequence[str],
    labels: Sequence[str],
    grid: Iterable[float] = tuple(np.round(np.linspace(0, 1, 21), 2)),
) -> tuple[float, float]:
    """Grid-search the weight of the first model in a two-model blend by macro F1."""
    first, second = probas
    label_array = np.asarray(labels)
    best_weight, best_f1 = 0.0, -1.0
    for weight in grid:
        predictions = label_array[(weight * first + (1 - weight) * second).argmax(axis=1)]
        score = f1_score(list(y_true), predictions, average="macro")
        if score > best_f1 + 1e-9:
            best_weight, best_f1 = float(weight), float(score)
    return best_weight, best_f1


class BlendedTicketModel:
    """Weighted soft-vote over already-fitted component models (sklearn-style API)."""

    def __init__(self, components: dict[str, Any], weights: dict[str, float], labels: Sequence[str]):
        self.components = components
        self.weights = {name: float(weights[name]) for name in components}
        self.labels = [str(label) for label in labels]

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.labels)

    def bind(self, base_dir: str | Path) -> "BlendedTicketModel":
        for component in self.components.values():
            if hasattr(component, "bind"):
                component.bind(base_dir)
        return self

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        text_list = [str(text) for text in texts]
        total = sum(self.weights.values()) or 1.0
        output = np.zeros((len(text_list), len(self.labels)))
        for name, component in self.components.items():
            if self.weights[name] > 0:
                output += self.weights[name] * aligned_proba(component, text_list, self.labels)
        return output / total

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]
