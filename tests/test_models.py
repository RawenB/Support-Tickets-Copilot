from __future__ import annotations

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
