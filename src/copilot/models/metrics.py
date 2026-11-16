"""Evaluation metrics shared by training, evaluation and the CI gate.

Besides overall macro F1, every report contains a *novel-ticket* slice: test tickets whose
nearest training ticket (TF-IDF cosine) is below ``novelty_threshold``. About half of the
test set are paraphrases of training tickets, so overall F1 alone rewards memorisation;
the novel slice shows how well a model generalises to tickets it has never seen.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def novelty_scores(train_texts: Sequence[str], eval_texts: Sequence[str]) -> np.ndarray:
    """Max cosine similarity of each eval text to any training text (0 = brand new)."""
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(train_texts)
    train_matrix = vectorizer.transform(train_texts)
    eval_matrix = vectorizer.transform(eval_texts)
    best = np.zeros(eval_matrix.shape[0])
    for start in range(0, eval_matrix.shape[0], 1024):
        block = eval_matrix[start : start + 1024] @ train_matrix.T
        best[start : start + 1024] = block.max(axis=1).toarray().ravel()
    return best


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
    novel_mask: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    y_true = np.asarray([str(v) for v in y_true])
    y_pred = np.asarray([str(v) for v in y_pred])
    metrics: dict[str, Any] = {
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        f"{prefix}weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if novel_mask is not None and novel_mask.any():
        metrics[f"{prefix}novel_macro_f1"] = float(
            f1_score(y_true[novel_mask], y_pred[novel_mask], average="macro", zero_division=0)
        )
        metrics[f"{prefix}novel_accuracy"] = float(accuracy_score(y_true[novel_mask], y_pred[novel_mask]))
        metrics[f"{prefix}novel_share"] = float(novel_mask.mean())
        if (~novel_mask).any():
            metrics[f"{prefix}seen_macro_f1"] = float(
                f1_score(y_true[~novel_mask], y_pred[~novel_mask], average="macro", zero_division=0)
            )
    return metrics


def per_class_report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    return classification_report(
        [str(v) for v in y_true],
        [str(v) for v in y_pred],
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )


def confusion_rows(y_true: Sequence[str], y_pred: Sequence[str]) -> list[dict[str, str]]:
    """Row-per-prediction format used by ``dvc plots`` confusion templates."""
    return [{"actual": str(a), "predicted": str(p)} for a, p in zip(y_true, y_pred)]


def confusion_figure(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix([str(v) for v in y_true], [str(v) for v in y_pred], labels=list(labels))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    size = max(5, 0.7 * len(labels) + 2)
    figure, axis = plt.subplots(figsize=(size, size))
    axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if normalized[i, j] > 0.5 else "black")
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title(title)
    figure.tight_layout()
    return figure
