# Architecture

This repository is organized around a Python-first support intelligence workflow with data ingestion, model training, evaluation, retrieval, and operational monitoring.

## Components

- Data pipeline: raw ingestion, cleaning, dedupe, and splits
- Models: queue and priority classification plus explanation utilities
- RAG: FAQ index and grounded answer drafting
- Incidents: DBSCAN and alert simulation
- Storage: SQLite-backed repository and DVC-managed datasets
- UI: Streamlit app using `st.navigation`, a dark theme defined in
  `.streamlit/config.toml`, a shared design system in `app/ui/`, Material
  Symbols icons (bundled with Streamlit), page views in `app/views/`, and UI
  data services in `app/services/`
- MLOps: MLflow, DVC, GitHub Actions, Evidently, and governance gates
