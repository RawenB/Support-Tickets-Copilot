"""Single source of truth for the model input text, shared by training and serving.

Training, evaluation, the stream and the UI must all build the model input the same way,
otherwise the model sees a different distribution in production than in training
(train/serve skew).
"""

from __future__ import annotations

import math
import re
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def _as_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def build_text(subject: Any, body: Any) -> str:
    """Return ``"<subject> <body>"`` lower-cased with collapsed whitespace."""
    joined = f"{_as_text(subject).strip()} {_as_text(body).strip()}"
    return _WHITESPACE.sub(" ", joined).strip().lower()


def frame_texts(frame: Any) -> list[str]:
    """Build model inputs for every row of a DataFrame with subject/body (or text) columns."""
    if "subject" in frame.columns or "body" in frame.columns:
        subjects = frame["subject"] if "subject" in frame.columns else [""] * len(frame)
        bodies = frame["body"] if "body" in frame.columns else [""] * len(frame)
        return [build_text(subject, body) for subject, body in zip(subjects, bodies)]
    if "text" in frame.columns:
        return [build_text("", text) for text in frame["text"]]
    raise ValueError("Frame requires subject/body or text columns.")
