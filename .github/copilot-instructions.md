# Copilot instructions

## Project structure

- Use the src layout under `src/copilot`.
- Keep all application code in Python modules under `src/copilot`.
- Use configuration files in `configs/` and `params.yaml` instead of hard-coded values.
- Keep the Streamlit UI in `app/`. Views live in `app/views/` and are registered
  with `st.navigation`; shared UI primitives live in `app/ui/`; data access for
  the UI lives in `app/services/`.

## Engineering rules

- Add type hints to all Python functions and methods.
- Use logging via the shared logger instead of `print`.
- Use a fixed random seed from `params.yaml` or config files everywhere training, sampling, and preprocessing is done.
- Treat `macro F1` as the primary model quality metric for queue and priority tasks.
- Keep all training and evaluation runs tied to MLflow: params, metrics, artifacts, git SHA, and DVC data hash.
- The support agent remains the final decision-maker. No model or workflow is allowed to send actions automatically.
- Every module should have corresponding test coverage.
- Prefer configuration-driven pipelines instead of embedded literals.

## Data and model governance

- Data is versioned through DVC, not Git.
- Use deterministic preprocessing and explicit data splits.
- Record the git SHA and dataset version when logging model metadata.
- Gate deployments on review and model quality thresholds.
