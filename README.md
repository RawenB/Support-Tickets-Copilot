# Support Copilot

Support Copilot is an internal support workspace for queue routing, priority triage, human review, grounded FAQ drafts, incident detection, and model governance.

## Prerequisites

- Python 3.9.6
- The raw CSV at `data/raw/customer-support-tickets.csv`
- The local virtual environment in `.venv`

## Run locally

```bash
source .venv/bin/activate
make data          # ingest, clean, dedupe, split
make rag           # build the FAQ index and evaluate retrieval
make train         # train queue and priority champions on the real splits
make evaluate      # write real test metrics
make stream        # replay the test split into SQLite
make drift         # write reports/drift_report.json
make app           # launch Streamlit on http://localhost:8501
```

Training requires real model artifacts. If a model artifact is unavailable, the stream stores predictions as `unclassified` or `very_low` with zero confidence instead of inventing labels.

## Useful commands

```bash
make lint
make test
make mlflow-ui
make retrain TASK=queue
make promote TASK=queue
```

`make demo` starts the UI with `COPILOT_DEMO_MODE=1`, which only adds an explicit demo label. It does not invent tickets or fall back to demo records.

## Product behavior

- SQLite is the product ticket source. The app never invents fallback tickets.
- Every draft is generated from retrieved FAQ passages and cites real FAQ IDs.
- Human review is mandatory; the agent confirms every outbound reply.
- The incidents page uses DBSCAN clusters and a rolling baseline, not hard-coded queue buckets.
- Admin views read real champion metadata, test metrics, data audits, and drift reports.
- Demo fallbacks were removed from the product path.

## Repo layout

- `app/` — Streamlit UI, navigation, and service layer
- `src/copilot/` — data pipeline, models, RAG, incidents, monitoring, repository
- `data/` — raw data, DVC-tracked splits, FAQ corpus, SQLite
- `configs/` — routing, training, incident, and RAG configuration
- `reports/` — real data, model, RAG, drift, and incident artifacts
- `docs/` — architecture, data card, model cards, and MLOps notes

## Verification

```bash
ruff check .
pytest -q
```

Quality gates are configured in `params.yaml` and enforced by `.github/workflows/model-gate.yml`.
