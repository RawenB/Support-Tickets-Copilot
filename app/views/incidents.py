"""Incident and live alert workspace."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.services.ticket_service import (
    get_active_tickets,
    priority_distribution,
    queue_distribution,
)
from app.ui.components import (
    alert_banner,
    empty_state,
    kpi_row,
    page_header,
    priority_badge,
    section_title,
    status_pill,
)


def _cluster_queues(tickets: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group tickets whose queue shows a possible incident signature."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for ticket in tickets:
        if str(ticket.get("priority", "")).lower() in {"critical", "high"} or str(
            ticket.get("status", "")
        ).lower() == "escalated":
            buckets.setdefault(str(ticket.get("queue", "general")).lower(), []).append(ticket)
    return {queue: items for queue, items in buckets.items() if len(items) >= 2}


@st.fragment(run_every="3s")
def incidents_view() -> None:
    """Display the active incident watch dashboard with alerting details."""
    tickets = get_active_tickets()
    escalated = [
        ticket
        for ticket in tickets
        if str(ticket.get("status", "")).lower() == "escalated"
    ]
    high_priority = [
        ticket
        for ticket in tickets
        if str(ticket.get("priority", "")).lower() in {"critical", "high"}
    ]
    clusters = _cluster_queues(tickets)

    page_header(
        "Incidents",
        "Spot ticket spikes early, track escalations, and keep the queue stable.",
        eyebrow="Reliability watch",
        icon="incidents",
    )

    kpi_row(
        [
            {
                "label": "Escalated",
                "value": len(escalated),
                "delta": "Open escalations",
                "tone": "danger" if escalated else "success",
                "icon": "escalate",
            },
            {
                "label": "High priority",
                "value": len(high_priority),
                "delta": "Critical + high severity",
                "tone": "warning" if high_priority else "success",
                "icon": "warning",
            },
            {
                "label": "Suspected clusters",
                "value": len(clusters),
                "delta": "Queues with spike signatures",
                "tone": "danger" if clusters else "success",
                "icon": "trend",
            },
            {
                "label": "Monitored tickets",
                "value": len(tickets),
                "delta": "Live stream window",
                "tone": "accent",
                "icon": "inbox",
            },
        ]
    )

    st.write("")
    if clusters:
        names = ", ".join(queue.title() for queue in clusters)
        alert_banner(
            f"Possible incident cluster detected in: {names}. "
            "Review the escalated queue before replying individually.",
            tone="danger",
        )
    else:
        alert_banner(
            "No active incident clusters detected. Live monitoring is stable.",
            tone="warning",
        )

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        with st.container(border=True):
            section_title("Escalated tickets", f"{len(escalated)} open", icon="escalate")
            if not escalated:
                empty_state(
                    "No escalations right now",
                    "Escalated tickets will appear here with owner and queue context.",
                )
            for ticket in escalated:
                st.markdown(
                    f"{priority_badge(str(ticket.get('priority', 'medium')))} "
                    f"{status_pill(str(ticket.get('status', 'escalated')))} "
                    f"**#{ticket.get('ticket_id')}** "
                    f"{ticket.get('subject', 'Untitled ticket')}",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"{ticket.get('queue', 'general').title()} · "
                    f"{ticket.get('customer_name', 'Unknown')} · "
                    f"assigned to {ticket.get('assigned_agent', 'Unassigned')}"
                )
                st.divider()

        with st.container(border=True):
            section_title(
                "High priority queue", "Critical and high severity", icon="warning"
            )
            if not high_priority:
                st.caption("No high-priority tickets in the current window.")
            for ticket in high_priority[:6]:
                st.markdown(
                    f"**#{ticket.get('ticket_id')}** "
                    f"{ticket.get('subject', 'Untitled ticket')}  \n"
                    f"<span class='muted' style='font-size:0.72rem;'>"
                    f"{ticket.get('queue', 'general').title()} · "
                    f"{str(ticket.get('status', 'new')).replace('_', ' ').title()}</span>",
                    unsafe_allow_html=True,
                )

    with right:
        queue_counts = queue_distribution(tickets)
        with st.container(border=True):
            section_title("Queue load", icon="queue")
            if queue_counts:
                frame = pd.DataFrame(
                    {"Queue": list(queue_counts), "Tickets": list(queue_counts.values())}
                ).set_index("Queue")
                st.bar_chart(frame, color="#7e9cc4", height=210)
            else:
                st.caption("No queue activity to chart.")

        priority_counts = priority_distribution(tickets)
        with st.container(border=True):
            section_title("Priority mix", icon="trend")
            for priority in ["critical", "high", "medium", "low"]:
                count = priority_counts.get(priority, 0)
                st.markdown(
                    f"{priority_badge(priority)} "
                    f"<span class='muted' style='font-size:0.78rem;'>{count} tickets</span>",
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            section_title("Detection rule", icon="shield")
            st.caption(
                "A queue is flagged when it contains at least two critical, high, "
                "or escalated tickets inside the live monitoring window."
            )
