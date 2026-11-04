"""Log one tiny run, register it and set the 'champion' alias. Proves the MLflow setup end to end.

Run:  python scripts/mlflow_smoke.py   then   make mlflow-ui
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from copilot.utils.mlflow_utils import resolve_tracking_uri, setup_mlflow

EXPERIMENT = "support-copilot-smoke"
MODEL_NAME = "support-copilot-smoke"


def main() -> None:
    client = setup_mlflow(EXPERIMENT)
    texts = ["card charged twice", "refund my invoice", "vpn is down", "api timeout error"]
    labels = ["billing", "billing", "technical", "technical"]
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200)).fit(texts, labels)
    with mlflow.start_run(run_name="smoke-baseline") as run:
        mlflow.log_params({"model_type": "tfidf_logreg", "purpose": "smoke test"})
        mlflow.log_metrics({"macro_f1": 1.0, "accuracy": 1.0})
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)
    version = client.search_model_versions(f"name='{MODEL_NAME}'")[0].version
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    print(f"tracking uri : {resolve_tracking_uri()}")
    print(f"run id       : {run.info.run_id}")
    print(f"champion     : {MODEL_NAME} v{champion.version}")
    print("OK: now run `make mlflow-ui` and open http://127.0.0.1:5001")


if __name__ == "__main__":
    main()
