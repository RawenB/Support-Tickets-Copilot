"""My activity workspace for handling metrics and recent decisions."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from app.services.ticket_service import (
    get_active_tickets,
    get_dashboard_metrics,
    get_recent_decisions,
    get_review_threshold,
    queue_distribution,
)
from app.ui.components import empty_state, kpi_row, page_header, section_title


@st.fragment(run_every="3s")
def activity_view() -> None:
    """Display handling metrics and recent human decisions for support agents."""
    review_threshold = get_review_threshold()
    tickets = get_active_tickets()
    metrics = get_dashboard_metrics()
    decisions = get_recent_decisions(limit=12)

    page_header(
        "My activity",
        "Understand your workload, review decisions, and spot queue pressure.",
        eyebrow="Personal performance",
        icon="activity",
    )

    kpi_row(
        [
            {
                "label": "Active tickets",
                "value": metrics["total"],
                "delta": f"{metrics['open']} open",
                "tone": "accent",
                "icon": "inbox",
            },
            {
                "label": "Needs review",
                "value": metrics["needs_review"],
                "delta": f"Below {review_threshold:.0%} confidence",
                "tone": "warning" if metrics["needs_review"] else "success",
                "icon": "review",
            },
            {
                "label": "Escalated",
                "value": metrics["escalated"],
                "delta": "Open escalations",
                "tone": "danger" if metrics["escalated"] else "success",
                "icon": "incidents",
            },
            {
                "label": "Avg confidence",
                "value": f"{float(metrics['avg_confidence']):.0%}",
                "delta": "Across active queue",
                "tone": "success",
                "icon": "trend",
            },
        ]
    )

    st.write("")
    left, right = st.columns([1.6, 1], gap="large")

    with left:
        with st.container(border=True):
            section_title("Queue load", "Where the work is concentrated", icon="queue")
            counts = queue_distribution(tickets)
            if counts:
                frame = pd.DataFrame(
                    {"Queue": list(counts), "Tickets": list(counts.values())}
                ).set_index("Queue")
                st.bar_chart(frame, color="#7e9cc4", height=230)
            else:
                st.caption("No queue activity to chart.")

        with st.container(border=True):
            section_title("Recent decisions", "Latest human actions", icon="history")
            if not decisions:
                empty_state(
                    "No decisions logged yet",
                    "Accept, edit, reject, or escalate a draft in the Inbox to build history.",
                )
            else:
                frame = pd.DataFrame(
                    [
                        {
                            "Ticket": f"#{row.get('ticket_id')}",
                            "Action": str(row.get("action", "")).title(),
                            "Agent": row.get("agent", "agent"),
                            "When": str(row.get("created_at", ""))[:19].replace("T", " "),
                        }
                        for row in decisions
                    ]
                )
                st.dataframe(frame, width="stretch", hide_index=True)

    with right:
        with st.container(border=True):
            section_title("Agent workload", "Assigned tickets", icon="agents")
            assignments = Counter(
                str(ticket.get("assigned_agent", "Unassigned")) for ticket in tickets
            )
            if assignments:
                frame = pd.DataFrame(
                    {
                        "Agent": list(assignments),
                        "Tickets": list(assignments.values()),
                    }
                ).sort_values("Tickets", ascending=False)
                st.dataframe(frame, width="stretch", hide_index=True)
            else:
                st.caption("No assignments in the current window.")

        with st.container(border=True):
            section_title("Focus areas", icon="focus")
            review = [
                ticket
                for ticket in tickets
                if float(ticket.get("confidence", review_threshold)) < review_threshold
            ]
            if review:
                for ticket in review[:5]:
                    st.markdown(
                        f"**#{ticket.get('ticket_id')}** "
                        f"{ticket.get('subject', 'Untitled ticket')[:42]}",
                    )
            else:
                st.caption("No low-confidence drafts. Great work.")
