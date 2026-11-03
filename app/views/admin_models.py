"""Admin-only models and release page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from app.services.stream import trigger_spike
from app.ui.components import kpi_row, page_header, section_title
from app.ui.icons import material_ref

REPORTS_DIR = Path("reports")


def _load_metrics() -> dict[str, Any]:
    for name in ("release.json", "metrics_summary.json", "queue_metrics.json"):
        path = REPORTS_DIR / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {}


def admin_models_view() -> None:
    """Show model metadata, operating thresholds, and versioning details."""
    metrics = _load_metrics()
    macro_f1 = metrics.get("macro_f1")

    page_header(
        "Models & versions",
        "Champion model, release quality gates, and incident simulation controls.",
        eyebrow="Admin",
        icon="models",
    )

    kpi_row(
        [
            {
                "label": "Champion alias",
                "value": "support-priority-v3",
                "delta": "Production traffic",
                "tone": "accent",
                "icon": "models",
            },
            {
                "label": "Version",
                "value": "3.4.1",
                "delta": "Latest promoted",
                "tone": "default",
                "icon": "description",
            },
            {
                "label": "Macro F1",
                "value": f"{float(macro_f1):.2f}" if isinstance(macro_f1, (int, float)) else "—",
                "delta": "Primary quality metric",
                "tone": "success",
                "icon": "trend",
            },
            {
                "label": "Confidence gate",
                "value": "0.80",
                "delta": "Routes to human review below",
                "tone": "warning",
                "icon": "shield",
            },
        ]
    )

    st.write("")
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        with st.container(border=True):
            section_title("Release summary", icon="description")
            st.markdown(
                "- Champion model: `support-priority-v3`\n"
                "- Version: `3.4.1`\n"
                "- Calibration: stable across the last 14 days\n"
                "- Human review is required below the 0.80 confidence gate"
            )
            st.caption(
                "See docs/MODEL_CARDS.md for the release rationale and quality gate."
            )

        with st.container(border=True):
            section_title("Latest evaluation artifact", icon="data")
            if metrics:
                st.json(metrics)
            else:
                st.caption("No evaluation artifact found in reports/.")

    with right:
        with st.container(border=True):
            section_title("Incident simulation", "Admin only", icon="spike")
            st.caption(
                "Inject a synthetic ticket spike to validate incident detection "
                "and the review queue under load."
            )
            if st.button(
                "Simulate technical spike",
                type="primary",
                width="stretch",
                icon=material_ref("spike"),
            ):
                cluster = trigger_spike(queue="technical")
                st.success(f"Injected {len(cluster)} tickets into the incident stream.")

        with st.container(border=True):
            section_title("Governance gates", icon="shield")
            st.markdown(
                "- Macro F1 must clear the configured threshold before promotion\n"
                "- Every run is logged to MLflow with git SHA and DVC data hash\n"
                "- Agents remain the final decision-maker for all outbound replies"
            )
