"""Needs review queue for low-confidence tickets."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.services.ticket_service import (
    PRIORITY_ORDER,
    REVIEW_CONFIDENCE_THRESHOLD,
    format_ticket_age,
    get_active_tickets,
)
from app.ui.components import (
    confidence_bar,
    empty_state,
    kpi_row,
    page_header,
    priority_badge,
    status_pill,
)


@st.fragment(run_every="3s")
def needs_review_view() -> None:
    """Show the tickets that require human review before resolution."""
    tickets = get_active_tickets()
    review_tickets = sorted(
        (
            ticket
            for ticket in tickets
            if float(ticket.get("confidence", REVIEW_CONFIDENCE_THRESHOLD))
            < REVIEW_CONFIDENCE_THRESHOLD
        ),
        key=lambda item: (
            float(item.get("confidence", 1.0)),
            PRIORITY_ORDER.get(str(item.get("priority", "low")).lower(), 9),
        ),
    )

    page_header(
        "Needs review",
        "Low-confidence drafts that need a human decision before they reach a customer.",
        eyebrow="Quality queue",
        icon="review",
    )

    average_confidence = (
        round(
            sum(float(ticket.get("confidence", 0.0)) for ticket in review_tickets)
            / len(review_tickets),
            2,
        )
        if review_tickets
        else 0.0
    )
    critical_count = sum(
        1
        for ticket in review_tickets
        if str(ticket.get("priority", "")).lower() in {"critical", "high"}
    )
    kpi_row(
        [
            {
                "label": "Awaiting review",
                "value": len(review_tickets),
                "delta": f"{len(tickets)} active tickets",
                "tone": "warning" if review_tickets else "success",
                "icon": "review",
            },
            {
                "label": "Avg confidence",
                "value": f"{average_confidence:.0%}",
                "delta": f"Threshold {REVIEW_CONFIDENCE_THRESHOLD:.0%}",
                "tone": "danger" if average_confidence < REVIEW_CONFIDENCE_THRESHOLD else "success",
                "icon": "trend",
            },
            {
                "label": "High priority",
                "value": critical_count,
                "delta": "Critical or high severity",
                "tone": "danger" if critical_count else "success",
                "icon": "incidents",
            },
            {
                "label": "Coverage",
                "value": f"{(1 - len(review_tickets) / len(tickets)):.0%}" if tickets else "—",
                "delta": "Auto-cleared by model",
                "tone": "accent",
                "icon": "confirm",
            },
        ]
    )

    st.write("")
    if not review_tickets:
        empty_state(
            "Everything is cleared",
            "All current drafts meet the confidence threshold.",
            icon="confirm",
        )
        return

    queue_options = [
        "All",
        *sorted({str(ticket.get("queue", "general")) for ticket in review_tickets}),
    ]
    queue_filter = st.selectbox(
        "Queue",
        queue_options,
        key="review_queue_filter",
        label_visibility="collapsed",
    )
    if queue_filter != "All":
        review_tickets = [
            ticket
            for ticket in review_tickets
            if str(ticket.get("queue", "")).lower() == queue_filter.lower()
        ]

    columns = st.columns(2, gap="large")
    for index, ticket in enumerate(review_tickets[:12]):
        with columns[index % 2]:
            _review_card(ticket)


def _review_card(ticket: dict[str, Any]) -> None:
    confidence = float(ticket.get("confidence", 0.0))
    age = ticket.get("age") or format_ticket_age(ticket)
    with st.container(border=True):
        st.markdown(
            f"{priority_badge(str(ticket.get('priority', 'medium')))} "
            f"{status_pill(str(ticket.get('status', 'new')))}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**#{ticket.get('ticket_id')}** {ticket.get('subject', 'Untitled ticket')}"
        )
        st.caption(
            f"{ticket.get('customer_name', 'Unknown')} · "
            f"{ticket.get('queue', 'general').title()} · {age}"
        )
        confidence_bar(confidence, caption=f"Confidence {confidence:.0%}")
        st.caption("Open the Inbox to action this ticket and send a reviewed reply.")
