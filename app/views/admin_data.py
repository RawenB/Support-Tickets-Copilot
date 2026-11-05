"""Admin data quality and drift backlog."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.ticket_service import (
    get_active_tickets,
    get_data_audit,
    get_drift_report,
    get_review_threshold,
    priority_distribution,
    queue_distribution,
)
from app.ui.components import (
    alert_banner,
    kpi_row,
    page_header,
    priority_badge,
    section_title,
)


def admin_data_view() -> None:
    """Display real data audit, distribution, and drift artifacts."""
    review_threshold = get_review_threshold()
    tickets = get_active_tickets()
    audit = get_data_audit()
    drift = get_drift_report()
    review_count = sum(
        1
        for ticket in tickets
        if float(ticket.get("confidence", review_threshold)) < review_threshold
    )
    review_rate = (review_count / len(tickets) * 100) if tickets else 0.0
    drift_share = drift.get("share_of_drifted_columns")
    drift_detected = bool(drift.get("dataset_drift"))

    page_header(
        "Data & drift",
        "Dataset health, coverage, and real drift signals for the support model.",
        eyebrow="Admin",
        icon="data",
    )

    kpi_row(
        [
            {
                "label": "Raw records",
                "value": audit.get("raw_rows", "—"),
                "delta": "English support tickets",
                "tone": "accent",
                "icon": "database",
            },
            {
                "label": "Test rows",
                "value": (audit.get("splits") or {}).get("test", "—"),
                "delta": "Held-out evaluation split",
                "tone": "default",
                "icon": "description",
            },
            {
                "label": "Drift share",
                "value": f"{float(drift_share):.0%}" if isinstance(drift_share, (int, float)) else "—",
                "delta": drift.get("method", "no drift report"),
                "tone": "danger" if drift_detected else "success",
                "icon": "trend",
            },
            {
                "label": "Review rate",
                "value": f"{review_rate:.0f}%",
                "delta": "Below confidence gate",
                "tone": "warning" if review_rate > 25 else "success",
                "icon": "review",
            },
        ]
    )

    st.write("")
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        with st.container(border=True):
            section_title("Queue coverage", "Tickets per queue", icon="queue")
            counts = queue_distribution(tickets)
            if counts:
                frame = pd.DataFrame(
                    {"Queue": list(counts), "Tickets": list(counts.values())}
                ).set_index("Queue")
                st.bar_chart(frame, color="#7e9cc4", height=230)
            else:
                st.caption("No records available.")

        with st.container(border=True):
            section_title("Priority distribution", icon="trend")
            priority_counts = priority_distribution(tickets)
            for priority in ["critical", "high", "medium", "low", "very_low"]:
                st.markdown(
                    f"{priority_badge(priority)} "
                    f"<span class='muted' style='font-size:0.78rem;'>"
                    f"{priority_counts.get(priority, 0)} tickets</span>",
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            section_title("Drift by feature", "Evidently data drift preset", icon="data")
            columns = drift.get("drift_by_columns") or {}
            if columns:
                rows = [
                    {
                        "Feature": feature,
                        "Score": details.get("score", details.get("psi", "—")),
                        "Threshold": details.get("threshold", "—"),
                        "Drift": "yes" if details.get("drift_detected") else "no",
                    }
                    for feature, details in columns.items()
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.caption("No drift artifact found. Run `python -m copilot.monitoring.drift`.")

    with right:
        if drift_detected:
            alert_banner(
                "Data drift detected. Review the affected features before promoting "
                "a model or retraining.",
                tone="danger",
            )
        else:
            alert_banner(
                "No dataset-level drift detected in the latest report.",
                tone="success",
            )
        st.write("")
        with st.container(border=True):
            section_title("Data governance", icon="shield")
            st.markdown(
                f"- Raw rows: `{audit.get('raw_rows', 'n/a')}`\n"
                f"- English rows: `{audit.get('english_rows', 'n/a')}`\n"
                f"- Duplicate rows removed: `{audit.get('duplicate_rows', 'n/a')}`\n"
                f"- Null rows removed: `{audit.get('null_rows', 'n/a')}`\n"
                "- Data versioning is managed through DVC, not Git\n"
                "- Splits are deterministic and recorded with the run"
            )
