"""Generate model explanations and feature-level diagnostics for triage decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _linear_model(model: Any) -> Any:
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("clf")
        if hasattr(classifier, "calibrated_classifiers_") and classifier.calibrated_classifiers_:
            return classifier.calibrated_classifiers_[0].estimator
        return classifier
    return model


def explain(
    text: str | None = None,
    *,
    task: str = "queue",
    model_path: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Return top linear features globally or for one ticket."""
    path = (
        Path(model_path)
        if model_path
        else Path("models") / f"{task}_model" / f"{task}_model.joblib"
    )
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    model = joblib.load(path)
    vectorizer = model.named_steps.get("tfidf") if hasattr(model, "named_steps") else None
    classifier = _linear_model(model)
    if vectorizer is None or not hasattr(classifier, "coef_"):
        return {
            "task": task,
            "model_type": type(model).__name__,
            "available": False,
            "reason": "Feature attribution is only implemented for linear text models.",
        }

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = np.asarray(classifier.coef_)
    labels = [str(label) for label in getattr(classifier, "classes_", [])]
    if text:
        vector = vectorizer.transform([text]).toarray()[0]
        scores = coefficients @ vector
        contributions = scores * vector
        order = np.argsort(np.abs(contributions))[::-1][:top_k]
        return {
            "task": task,
            "text": text,
            "available": True,
            "features": [
                {
                    "feature": str(feature_names[index]),
                    "contribution": float(contributions[index]),
                    "value": float(vector[index]),
                }
                for index in order
                if vector[index] > 0
            ],
        }

    global_terms: dict[str, list[dict[str, float]]] = {}
    for index, label in enumerate(labels):
        order = np.argsort(coefficients[index])[::-1][:top_k]
        global_terms[label] = [
            {"feature": str(feature_names[item]), "weight": float(coefficients[index][item])}
            for item in order
        ]
    return {"task": task, "available": True, "top_features": global_terms}
