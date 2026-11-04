"""Honest-metrics contract. Skipped until the models have been trained on the real splits."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())
TASKS = ("queue", "priority")


def _metrics(task: str) -> dict:
    path = ROOT / "reports" / f"metrics_{task}.json"
    if not path.exists():
        pytest.skip(f"{path.name} missing: train the {task} model first")
    return json.loads(path.read_text())


@pytest.mark.parametrize("task", TASKS)
def test_metrics_file_has_the_required_fields(task):
    m = _metrics(task)
    for key in ("task", "split", "model_type", "macro_f1", "accuracy", "n_test", "mlflow_run_id"):
        assert key in m, f"metrics_{task}.json is missing '{key}'"
    assert m["split"] == "test"


@pytest.mark.parametrize("task", TASKS)
def test_metrics_are_computed_on_a_real_sized_test_set(task):
    assert _metrics(task)["n_test"] >= 1000


@pytest.mark.parametrize("task", TASKS)
def test_f1_is_believable_not_perfect(task):
    f1 = _metrics(task)["macro_f1"]
    assert 0.20 <= f1 < 0.999, f"macro F1={f1}: suspiciously low or perfect (leakage or toy data)"


@pytest.mark.parametrize("task", TASKS)
def test_champion_meets_the_gate(task):
    assert _metrics(task)["macro_f1"] >= PARAMS["f1_gate"], "champion below the F1 gate"


def test_predict_returns_label_and_confidence():
    from copilot.models.predict import predict

    out = predict("API calls are timing out for several users", task="queue")
    assert isinstance(out["label"], str) and out["label"]
    assert 0.0 < out["confidence"] <= 1.0
    batch = predict(["I was charged twice this month", "VPN router is down"], task="queue")
    assert len(batch) == 2 and all("confidence" in b for b in batch)


def test_predict_is_deterministic():
    from copilot.models.predict import predict

    text = "Please reset my password, I cannot log in"
    assert predict(text, task="queue") == predict(text, task="queue")


def test_mlflow_has_runs_and_a_champion_alias():
    mlflow = pytest.importorskip("mlflow")
    uri = PARAMS.get("mlflow", {}).get("tracking_uri")
    if not uri:
        pytest.skip("params.yaml has no mlflow.tracking_uri yet")
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    for task in TASKS:
        experiment = client.get_experiment_by_name(f"support-copilot-{task}")
        assert experiment is not None, f"no MLflow experiment for {task}"
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) >= 3, f"{task}: need >= 3 tracked runs (baseline + DistilBERT variants)"
        types = {r.data.params.get("model_type") for r in runs}
        assert len(types) >= 2, f"{task}: compare at least two model types, got {types}"
        champion = client.get_model_version_by_alias(f"support-copilot-{task}", "champion")
        assert champion is not None
