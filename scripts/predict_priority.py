from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "baseline_priority_model.joblib"


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    title = str(payload.get("title", "") or "").strip()
    description = str(payload.get("description", "") or "").strip()
    text = " ".join(part for part in [title, description] if part)

    if not text:
        print(json.dumps({"priority": "medium", "confidence": 0.0, "source": "empty"}))
        return

    model = joblib.load(MODEL_PATH)
    predicted = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))

    print(
        json.dumps(
            {
                "priority": predicted.lower(),
                "confidence": round(confidence, 4),
                "source": "model",
            }
        )
    )


if __name__ == "__main__":
    main()
