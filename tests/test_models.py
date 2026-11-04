from __future__ import annotations

from pathlib import Path

import pandas as pd

from copilot.data.audit import audit
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


def test_baseline_and_audit_exercise_real_pipeline_behavior(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject": ["login issue", "login issue", "refund request", "refund request"],
            "body": ["cannot sign in", "failed sign in", "charged wrong", "refund money"],
            "queue": ["technical", "technical", "billing", "billing"],
            "priority": ["high", "high", "low", "low"],
            "language": ["en", "en", "en", "en"],
        }
    )
    baseline_result = baseline(df, task="queue")
    assert baseline_result["task"] == "queue"
    assert baseline_result["macro_f1"] >= 0.5

    audit_summary = audit(df)
    assert audit_summary["row_count"] == 4
    assert audit_summary["languages"]["en"] == 4

    prediction = predict(
        "cannot access my account",
        task="queue",
        model_path=baseline_result["model_path"],
    )
    assert prediction["task"] == "queue"
    assert prediction["label"] in {"technical", "billing"}

    calibrated = calibrate(baseline_result["model_path"], task="queue")
    assert "threshold" in calibrated
    assert calibrated["threshold"] > 0.0
