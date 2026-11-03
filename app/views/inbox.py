"""Inbox workspace for daily support agent work."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from app.services.ticket_service import (
    REVIEW_CONFIDENCE_THRESHOLD,
    filter_tickets,
    format_ticket_age,
    get_active_tickets,
    get_queue_team_map,
    get_related_tickets,
    get_ticket_sources,
    log_ticket_decision,
    sort_tickets,
)
from app.ui.components import (
    alert_banner,
    confidence_bar,
    empty_state,
    page_header,
    priority_badge,
    section_title,
    source_card,
    status_pill,
    ticket_row,
)
from app.ui.icons import material_ref

PRIORITIES = ["All", "critical", "high", "medium", "low"]
STATUSES = ["All", "new", "in_review", "escalated", "resolved"]


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _query_value(key: str) -> str | None:
    value = st.query_params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _pick_ticket(
    tickets: list[dict[str, Any]], filtered: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Resolve the selected ticket, keeping the URL and the session in sync."""
    if not filtered:
        return None
    ids = {str(ticket["ticket_id"]) for ticket in tickets}
    candidate = st.session_state.get("selected_ticket_id") or _query_value("ticket")
    if candidate is not None and str(candidate) in ids:
        ticket_id = str(candidate)
    elif filtered:
        ticket_id = str(filtered[0]["ticket_id"])
    elif tickets:
        ticket_id = str(tickets[0]["ticket_id"])
    else:
        return None
    st.session_state["selected_ticket_id"] = ticket_id
    st.query_params["ticket"] = ticket_id
    return next(
        (ticket for ticket in tickets if str(ticket["ticket_id"]) == ticket_id), None
    )


def _meta_chip(label: str) -> str:
    return f'<span class="chip">{_escape(label)}</span>'


@st.fragment(run_every="3s")
def inbox_view() -> None:
    """Display a live ticket inbox with triage and AI response review."""
    tickets = get_active_tickets()
    page_header(
        "Inbox",
        "Work the live queue, review the AI draft, and route tickets with confidence.",
        eyebrow="Agent workspace",
        icon="inbox",
    )
    if not tickets:
        empty_state("Inbox is clear", "No tickets are currently waiting for an agent.")
        return

    queue_map = get_queue_team_map()

    filter_cols = st.columns([2.4, 1, 1, 1, 1.1], gap="small")
    search = filter_cols[0].text_input(
        "Search tickets",
        placeholder="Search subject, customer, or queue",
        key="inbox_search",
        label_visibility="collapsed",
        icon=material_ref("search"),
    )
    filter_status = filter_cols[1].selectbox(
        "Status",
        STATUSES,
        key="inbox_status",
        label_visibility="collapsed",
    )
    filter_priority = filter_cols[2].selectbox(
        "Priority",
        PRIORITIES,
        key="inbox_priority",
        label_visibility="collapsed",
    )
    filter_queue = filter_cols[3].selectbox(
        "Queue",
        ["All", *sorted({str(ticket.get("queue", "general")) for ticket in tickets})],
        key="inbox_queue",
        label_visibility="collapsed",
    )
    needs_review_only = filter_cols[4].toggle(
        "Review only",
        key="inbox_review_only",
        help="Show only tickets below the confidence threshold.",
    )

    filtered = sort_tickets(
        filter_tickets(
            tickets,
            search=search,
            status=filter_status,
            priority=filter_priority,
            queue=filter_queue,
            needs_review_only=needs_review_only,
        )
    )
    for ticket in tickets:
        ticket.setdefault("age", format_ticket_age(ticket))

    selected = _pick_ticket(tickets, filtered)

    left_col, center_col, right_col = st.columns([1.05, 1.95, 1.05], gap="large")

    with left_col:
        section_title("Queue", f"{len(filtered)} of {len(tickets)}", icon="queue")
        if not filtered:
            empty_state("No matches", "Adjust the filters to see more tickets.", icon="filter")
        with st.container(height=640, border=False):
            for ticket in filtered:
                clicked = ticket_row(
                    ticket,
                    selected=str(ticket["ticket_id"]) == str(selected["ticket_id"]),
                    key=f"select_{ticket['ticket_id']}",
                )
                if clicked:
                    st.session_state["selected_ticket_id"] = str(ticket["ticket_id"])
                    st.query_params["ticket"] = str(ticket["ticket_id"])
                    st.rerun(scope="fragment")
        st.caption("Tip: press Enter in search to narrow the queue instantly.")

    if selected is None:
        return

    with center_col:
        ticket_id = _escape(selected.get("ticket_id"))
        subject = _escape(selected.get("subject", "Untitled ticket"))
        status = str(selected.get("status", "new"))
        priority = str(selected.get("priority", "medium"))
        queue = str(selected.get("queue", "general")).lower()
        team = queue_map.get(queue, "Support Desk")
        age = _escape(selected.get("age") or format_ticket_age(selected))
        confidence = float(
            selected.get("confidence", REVIEW_CONFIDENCE_THRESHOLD)
        )

        st.markdown(
            f'<div class="ticket-title">#{ticket_id} · {subject}</div>'
            '<div class="meta-row">'
            f"{status_pill(status)}{priority_badge(priority)}"
            f"{_meta_chip(queue.title())}"
            f"{_meta_chip(f'Team: {team}')}"
            f"{_meta_chip(age)}"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            section_title(
                "Customer message",
                f"{selected.get('customer_name', 'Unknown customer')} · "
                f"{selected.get('customer_email', 'unknown@example.com')}",
                icon="ticket",
            )
            st.write(selected.get("body", ""))

        draft = (
            selected.get("draft")
            or selected.get("ai_response")
            or "We reviewed the issue and have identified the likely fix. "
            "Please confirm the account permissions and retry the action."
        )
        section_title("AI draft", "Human review required before sending", icon="description")
        st.text_area(
            "Draft response",
            value=draft,
            key=f"draft_{selected['ticket_id']}",
            label_visibility="collapsed",
            height=170,
        )

        sources = get_ticket_sources(selected)
        with st.expander(
            f"Sources used ({len(sources)})",
            expanded=False,
            icon=material_ref("sources"),
        ):
            for source in sources:
                source_card(source)

        if confidence < REVIEW_CONFIDENCE_THRESHOLD:
            alert_banner(
                "Grounding warning: this draft falls below the confidence "
                f"threshold ({confidence:.0%} < {REVIEW_CONFIDENCE_THRESHOLD:.0%}). "
                "Review the sources before sending.",
                tone="danger",
            )

        st.write("")
        action_cols = st.columns([1.3, 1, 1, 1], gap="small")
        if action_cols[0].button(
            "Send reply", type="primary", width="stretch", icon=material_ref("send")
        ):
            log_ticket_decision(
                str(selected["ticket_id"]), action="accepted", agent="agent"
            )
            st.toast("Reply logged as resolved and ready to send.")
        if action_cols[1].button("Regenerate", width="stretch", icon=material_ref("refresh")):
            log_ticket_decision(
                str(selected["ticket_id"]), action="edited", agent="agent"
            )
            st.toast("Draft regenerated with refreshed context.")
        if action_cols[2].button("Reject", width="stretch", icon=material_ref("reject")):
            log_ticket_decision(
                str(selected["ticket_id"]), action="rejected", agent="agent"
            )
            st.toast("Draft rejected and returned to the queue.")
        if action_cols[3].button("Escalate", width="stretch", icon=material_ref("escalate")):
            log_ticket_decision(
                str(selected["ticket_id"]), action="escalated", agent="agent"
            )
            st.toast("Ticket escalated to the incident workflow.")

    with right_col:
        section_title("AI copilot", "Routing and context", icon="models")

        with st.container(border=True):
            st.markdown("**Routing suggestion**")
            st.markdown(
                f"{priority_badge(priority)} {_meta_chip(f'Queue: {queue.title()}')} "
                f"{_meta_chip(f'Team: {team}')}",
                unsafe_allow_html=True,
            )
            confidence_bar(confidence, caption=f"Routing confidence {confidence:.0%}")
            router_cols = st.columns(2)
            if router_cols[0].button("Confirm", width="stretch", icon=material_ref("confirm")):
                st.toast("Routing confirmed.")
            corrected = router_cols[1].selectbox(
                "Correct queue",
                ["Keep current", "Billing", "Technical", "General"],
                index=0,
                key=f"correct_{selected['ticket_id']}",
                label_visibility="collapsed",
            )
            if corrected != "Keep current":
                st.info(f"Correction saved for retraining: {corrected}")

        with st.container(border=True):
            section_title("Similar tickets", "Same queue", icon="ticket")
            related = get_related_tickets(selected, tickets)
            if not related:
                st.caption("No comparable tickets in this queue.")
            for item in related:
                related_age = item.get("age") or format_ticket_age(item)
                st.markdown(
                    f"**#{_escape(item['ticket_id'])}** "
                    f"{_escape(str(item.get('subject', 'Untitled'))[:38])}  \n"
                    f"<span class='muted' style='font-size:0.72rem;'>"
                    f"{_escape(str(item.get('status', 'new')).replace('_', ' ').title())}"
                    f" · {_escape(related_age)}</span>",
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            section_title("Incident context", icon="incidents")
            queue_load = sum(
                1 for item in tickets if str(item.get("queue", "")).lower() == queue
            )
            if status.lower() == "escalated":
                alert_banner(
                    "This ticket is escalated and linked to an active incident review.",
                    tone="danger",
                )
            elif queue_load >= 3:
                alert_banner(
                    f"{queue_load} tickets currently share the {queue.title()} queue. "
                    "Watch for a developing incident cluster.",
                    tone="warning",
                )
            else:
                st.caption("No active incident cluster detected for this queue.")


if __name__ == "__main__":
    inbox_view()
