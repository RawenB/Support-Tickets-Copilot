"""One place to configure MLflow so training, the registry and the UI all read the same store."""

from __future__ import annotations

from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _mlflow_params() -> dict[str, str]:
    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8")) or {}
    cfg = params.get("mlflow", {})
    return {
        "tracking_uri": str(cfg.get("tracking_uri", "sqlite:///mlflow.db")),
        "artifact_root": str(cfg.get("artifact_root", "./mlartifacts")),
    }


def resolve_tracking_uri() -> str:
    """Return an absolute tracking URI so the result never depends on the working directory."""
    uri = _mlflow_params()["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return f"{prefix}{(PROJECT_ROOT / uri[len(prefix):]).resolve()}"
    return uri


def resolve_artifact_root() -> Path:
    root = Path(_mlflow_params()["artifact_root"])
    return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()


def setup_mlflow(experiment: str) -> MlflowClient:
    """Point MLflow at the project store and make sure the experiment exists."""
    mlflow.set_tracking_uri(resolve_tracking_uri())
    client = MlflowClient()
    if client.get_experiment_by_name(experiment) is None:
        artifact_dir = resolve_artifact_root() / experiment
        artifact_dir.mkdir(parents=True, exist_ok=True)
        client.create_experiment(experiment, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(experiment)
    return client
