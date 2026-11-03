# Support Copilot

Support Copilot is the internal support workspace for agents and admins. It brings together queue routing, human review, incident detection, and a polished inbox workflow in one product surface.

## Run locally

1. Start the stream replay worker:
   python -m copilot.incidents.simulate_stream
2. Launch the app:
   streamlit run app/streamlit_app.py

## Product behavior

- Agent-first inbox with triage, review, and resolution actions.
- SQLite-backed ticket source with demo fallback when model artifacts are unavailable.
- Admin-only views for models, drift, and release checks.
- No automatic send actions: the agent must explicitly confirm any reply.

## Repo layout

- app/ — Streamlit product UI and services
- app/views/ — navigation pages registered with `st.navigation`
- app/ui/ — shared design system, layout shell, and reusable components
- .streamlit/ — dark theme configuration; icons use the Material Symbols font bundled with Streamlit
- app/services/ — data access, analytics, and demo-mode fallbacks for the UI
- src/copilot/ — business logic, repository, routing, incidents, and retrieval code
- data/ — raw, processed, and demo data
- configs/ — product configuration and routing labels
- docs/ — model cards and operational docs

## Verification

Use pytest and ruff before shipping changes:

python -m pytest
ruff check .
