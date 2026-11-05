"""Admin-only models and release page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.services.stream import trigger_spike
from app.services.ticket_service import (
    get_f1_gate,
    get_model_registry_status,
    get_review_threshold,
)
from app.ui.components import kpi_row, page_header, section_title
from app.ui.icons import material_ref


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def admin_models_view() -> None:
    """Show real champion metadata, test metrics, and governance gates."""
    status = get_model_registry_status()
    tasks = status.get("tasks", {})
    queue_task = tasks.get("queue", {})
    priority_task = tasks.get("priority", {})
    queue_metrics = queue_task.get("metrics", {})
    priority_metrics = priority_task.get("metrics", {})
    gate = get_f1_gate()
    review_threshold = get_review_threshold()

    page_header(
        "Models & versions",
        "Champion models, real test metrics, and promotion gates.",
        eyebrow="Admin",
        icon="models",
    )

    kpi_row(
        [
            {
                "label": "Queue macro F1",
                "value": (
                    f"{value:.2f}"
                    if (value := _metric_value(queue_metrics, "macro_f1")) is not None
                    else "—"
                ),
                "delta": "Real test split",
                "tone": "success",
                "icon": "trend",
            },
            {
                "label": "Priority macro F1",
                "value": (
                    f"{value:.2f}"
                    if (value := _metric_value(priority_metrics, "macro_f1")) is not None
                    else "—"
                ),
                "delta": "Real test split",
                "tone": "success",
                "icon": "trend",
            },
            {
                "label": "Champion alias",
                "value": "priority:champion",
                "delta": str(priority_task.get("alias", "not registered")),
                "tone": "accent",
                "icon": "models",
            },
            {
                "label": "Quality gate",
                "value": f"{gate:.2f}",
                "delta": f"Review below {review_threshold:.0%} confidence",
                "tone": "warning",
                "icon": "shield",
            },
        ]
    )

    st.write("")
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        with st.container(border=True):
            section_title("Champion summary", icon="description")
            if not queue_task and not priority_task:
                st.caption("No champion metadata found. Train the real models first.")
            for task_name, task in (("Queue", queue_task), ("Priority", priority_task)):
                champion = task.get("champion", {})
                if not champion:
                    st.markdown(f"**{task_name}**: not trained")
                    continue
                st.markdown(
                    f"**{task_name}**: `{champion.get('model_type', 'unknown')}` · "
                    f"test macro F1 `{float(champion.get('macro_f1', 0)):.3f}` · "
                    f"accuracy `{float(champion.get('accuracy', 0)):.3f}`"
                )
                st.caption(
                    f"Run `{champion.get('mlflow_run_id', 'n/a')}` · "
                    f"artifact `{champion.get('model_path', 'n/a')}`"
                )
            st.caption("See docs/MODEL_CARDS.md for the release rationale and limitations.")

        with st.container(border=True):
            section_title("Latest evaluation artifacts", icon="data")
            payload = {
                task: task_data.get("metrics", {})
                for task, task_data in tasks.items()
                if task_data.get("metrics")
            }
            if payload:
                st.json(payload)
            else:
                st.caption("No evaluation artifacts found in reports/.")

    with right:
        with st.container(border=True):
            section_title("Incident simulation", "Admin only", icon="spike")
            st.caption(
                "Inject a real-ticket spike into one queue to validate incident "
                "detection and the review queue under load."
            )
            if st.button(
                "Simulate technical spike",
                type="primary",
                width="stretch",
                icon=material_ref("spike"),
            ):
                cluster = trigger_spike(queue="technical_support")
                st.success(f"Injected {len(cluster)} tickets into the incident stream.")

        with st.container(border=True):
            section_title("Governance gates", icon="shield")
            st.markdown(
                f"- Macro F1 must clear `{gate:.2f}` before promotion\n"
                "- Promotion requires a registered MLflow champion alias\n"
                "- Every run logs parameters, metrics, git SHA, and DVC data hash\n"
                "- Agents remain the final decision-maker for all outbound replies"
            )
