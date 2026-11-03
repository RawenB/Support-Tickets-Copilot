from __future__ import annotations

from pathlib import Path

import pandas as pd

from copilot.models.baseline import baseline
from copilot.models.calibrate import calibrate
from copilot.models.evaluate import evaluate
from copilot.models.explain import explain
from copilot.models.predict import predict
from copilot.models.train import train


def test_model_modules_importable() -> None:
    assert callable(train)
    assert callable(baseline)
    assert callable(evaluate)
    assert callable(calibrate)
    assert callable(explain)
    assert callable(predict)


def test_train_and_evaluate_with_synthetic_data(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic_tickets.csv"
    pd.DataFrame(
        {
            "subject": [
                "cannot login",
                "password reset",
                "billing error",
                "refund request",
                "login issue",
                "reset password",
                "invoice problem",
                "money refund",
                "cannot access account",
                "account access issue",
                "charge disputed",
                "refund payment",
            ],
            "body": [
                "i cannot sign in",
                "i need reset",
                "overcharged invoice",
                "need refund",
                "sign in error",
                "password help",
                "invoice wrong",
                "refund money",
                "access blocked",
                "login blocked",
                "charge issue",
                "refund payment",
            ],
            "queue": [
                "technical",
                "technical",
                "billing",
                "billing",
                "technical",
                "technical",
                "billing",
                "billing",
                "technical",
                "technical",
                "billing",
                "billing",
            ],
            "priority": [
                "high",
                "high",
                "medium",
                "low",
                "high",
                "high",
                "medium",
                "low",
                "high",
                "high",
                "medium",
                "low",
            ],
            "language": ["en"] * 12,
        }
    ).to_csv(data_path, index=False)

    result = train(data_path=data_path, output_dir=tmp_path / "model_artifacts", task="queue")
    assert "model_path" in result
    assert result["task"] == "queue"
    assert result["macro_f1"] >= 0.0

    metrics = evaluate(data_path=data_path, model_path=result["model_path"], task="queue")
    assert "macro_f1" in metrics
    assert metrics["macro_f1"] >= 0.0
