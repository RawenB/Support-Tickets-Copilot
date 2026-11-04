"""Admin data quality and drift backlog."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.ticket_service import (
    get_active_tickets,
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
    """Display data drift, distribution, and dataset versioning context."""
    review_threshold = get_review_threshold()
    tickets = get_active_tickets()
    review_count = sum(
        1
        for ticket in tickets
        if float(ticket.get("confidence", review_threshold)) < review_threshold
    )
    review_rate = (review_count / len(tickets) * 100) if tickets else 0.0

    page_header(
        "Data & drift",
        "Dataset health, coverage, and drift signals for the support model.",
        eyebrow="Admin",
        icon="data",
    )

    kpi_row(
        [
            {
                "label": "Recorded tickets",
                "value": len(tickets),
                "delta": "Active dataset window",
                "tone": "accent",
                "icon": "database",
            },
            {
                "label": "Dataset version",
                "value": "v3",
                "delta": "customer-support-tickets",
                "tone": "default",
                "icon": "description",
            },
            {
                "label": "Drift score",
                "value": "Low",
                "delta": "Within configured bounds",
                "tone": "success",
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

    with right:
        alert_banner(
            "Drift report preview is available in the MLOps artifact store. "
            "Review it before promoting a release.",
            tone="warning",
        )
        st.write("")
        with st.container(border=True):
            section_title("Data governance", icon="shield")
            st.markdown(
                "- Data versioning is managed through DVC, not Git\n"
                "- Splits are deterministic and recorded with the run\n"
                "- Review low-confidence tickets before retraining\n"
                "- Confirm dataset freshness before the next model release"
            )
