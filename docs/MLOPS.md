# MLOps plan

This project uses DVC for data and pipeline versioning, MLflow for experiment tracking and model registration, GitHub Actions for CI validation, and a configured F1 gate before promotion.

## Reproducible pipeline

```bash
make data
make rag
make train
make evaluate
make stream
make drift
```

DVC stages:

- `ingest` → `clean` → `dedupe` → `split`
- `train_queue` / `train_priority` → real model artifacts and `reports/metrics_*.json`
- `evaluate` → `reports/metrics_summary.json`
- `build_faq_index` → `data/faq/index.faiss` and `data/faq/index_meta.json`
- `rag_eval` → `reports/rag_eval.json`
- `simulate_stream` → `data/interim/incident_stream.csv`
- `incident_eval` → `reports/incident_metrics.json`

## Experiment tracking

- Tracking URI: `sqlite:///mlflow.db`
- Artifact root: `./mlartifacts`
- Experiments: `support-copilot-queue`, `support-copilot-priority`, `support-copilot-rag`, `support-copilot-incidents`
- Registered model names: `support-copilot-queue`, `support-copilot-priority`
- Champion alias: `champion`

## Quality gate

`params.yaml` sets `f1_gate`. `make promote TASK=<queue|priority>` refuses to register a model below the gate. `.github/workflows/model-gate.yml` checks both real metrics files before promotion.

## Human feedback loop

1. Agents correct routing or priority in the UI.
2. Corrections are stored in SQLite.
3. `make retrain TASK=<queue|priority>` appends real corrections to the training split.
4. The retrain result is written to `reports/retrain_<task>.json`.
5. The model is promoted only when the retrained macro F1 clears the gate.

## Governance

- Support agents retain final decision authority.
- Model artifacts and metrics are versioned through MLflow and DVC.
- The product path never invents tickets or fallback predictions.
- Demo mode is explicit and only adds a label; it does not create data.
