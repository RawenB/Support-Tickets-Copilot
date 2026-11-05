"""Inbox workspace for daily support agent work."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from app.services.ticket_service import (
    confirm_routing,
    correct_routing,
    ensure_first_draft,
    escalate,
    filter_tickets,
    format_ticket_age,
    get_active_tickets,
    get_grounding_status,
    get_priority_choices,
    get_queue_choices,
    get_queue_team_map,
    get_related_tickets,
    get_review_threshold,
    get_ticket_sources,
    regenerate_draft,
    reject_draft,
    save_draft,
    send_reply,
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

PRIORITIES = ["All", "critical", "high", "medium", "low", "very_low"]
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


def _next_active_ticket_id(current_ticket_id: str) -> str | None:
    tickets = get_active_tickets()
    ordered_ids = [str(ticket["ticket_id"]) for ticket in tickets]
    if not ordered_ids:
        return None
    if current_ticket_id not in ordered_ids:
        return ordered_ids[0]
    current_index = ordered_ids.index(current_ticket_id)
    for candidate in ordered_ids[current_index + 1 :] + ordered_ids[:current_index]:
        if candidate != current_ticket_id:
            return candidate
    return ordered_ids[0] if ordered_ids[0] != current_ticket_id else None


def _run_action_and_refresh(ticket_id: str, callback: Any, *, advance: bool = True) -> Any:
    """Run a service action from an on_click callback (Streamlit reruns the fragment by itself).

    advance=True moves to the next ticket (use it for Send, Reject, Escalate).
    advance=False keeps the same ticket open (use it for Save edit and Regenerate).
    """
    result = callback()
    st.toast(result.message)
    if result.ok and advance:
        next_ticket_id = _next_active_ticket_id(str(ticket_id))
        if next_ticket_id:
            st.session_state["selected_ticket_id"] = str(next_ticket_id)
            st.query_params["ticket"] = str(next_ticket_id)
    return result


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
        confidence_threshold = get_review_threshold()
        confidence = float(
            selected.get("confidence", confidence_threshold)
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

        draft_key = f"draft_{selected['ticket_id']}"
        latest_draft = ensure_first_draft(str(selected["ticket_id"]))
        if draft_key not in st.session_state:
            st.session_state[draft_key] = str((latest_draft or {}).get("text", ""))
        section_title("AI draft", "Human review required before sending", icon="description")
        st.text_area(
            "Draft response",
            key=draft_key,
            label_visibility="collapsed",
            height=170,
            placeholder="No draft yet. Click Regenerate.",
        )
        if latest_draft:
            draft_source = str(latest_draft.get("source", "agent")).lower()
            label = {
                "rag": "retrieved FAQ passages",
                "template_fallback": "template (no FAQ match)",
                "agent": "edited by agent",
            }.get(draft_source, draft_source)
            st.caption(f"Draft version {latest_draft.get('version', 1)} · based on {label}")

        sources = get_ticket_sources(selected)
        with st.expander(
            f"Sources used ({len(sources)})",
            expanded=False,
            icon=material_ref("sources"),
        ):
            for source in sources:
                source_card(source)

        grounding = get_grounding_status(selected)
        if not grounding.get("grounded"):
            unsupported = grounding.get("unsupported_sentences") or []
            detail = (
                f" Unsupported: {unsupported[0][:140]}"
                if unsupported
                else f" Coverage: {grounding.get('coverage', 0):.0%}."
            )
            alert_banner(
                "Grounding warning: the draft may contain unsupported statements."
                + detail
                + " Review the sources before sending.",
                tone="danger",
            )
        elif confidence < confidence_threshold:
            alert_banner(
                "Confidence warning: this draft falls below the review "
                f"threshold ({confidence:.0%} < {confidence_threshold:.0%}). "
                "Review the sources before sending.",
                tone="danger",
            )

        st.write("")

        def _save_edit() -> None:
            draft_text = st.session_state.get(f"draft_{selected['ticket_id']}", "")
            _run_action_and_refresh(
                str(selected["ticket_id"]),
                lambda: save_draft(
                    str(selected["ticket_id"]),
                    text=draft_text,
                    source="agent",
                ),
                advance=False,
            )

        def _send_reply_action() -> None:
            draft_text = st.session_state.get(f"draft_{selected['ticket_id']}", "")
            _run_action_and_refresh(
                str(selected["ticket_id"]),
                lambda: send_reply(str(selected["ticket_id"]), text=draft_text, agent="agent"),
            )

        def _regenerate_action() -> None:
            result = _run_action_and_refresh(
                str(selected["ticket_id"]),
                lambda: regenerate_draft(str(selected["ticket_id"]), agent="agent"),
                advance=False,
            )
            if result.ok:
                # the text area is bound to session_state, so push the new text into it
                st.session_state[draft_key] = str(result.data["draft"]["text"])

        def _reject_action() -> None:
            _run_action_and_refresh(
                str(selected["ticket_id"]),
                lambda: reject_draft(
                    str(selected["ticket_id"]),
                    text=st.session_state.get(draft_key, ""),
                    agent="agent",
                ),
            )

        def _escalate_action() -> None:
            _run_action_and_refresh(
                str(selected["ticket_id"]),
                lambda: escalate(str(selected["ticket_id"]), agent="agent"),
            )

        action_cols = st.columns([1.4, 1.2, 1, 1, 1])
        action_cols[0].button(
            "Save edit",
            width="stretch",
            icon=material_ref("save"),
            on_click=_save_edit,
            disabled=status.lower() == "resolved",
        )
        action_cols[1].button(
            "Send reply",
            type="primary",
            width="stretch",
            icon=material_ref("send"),
            on_click=_send_reply_action,
            disabled=status.lower() == "resolved",
        )
        action_cols[2].button(
            "Regenerate",
            width="stretch",
            icon=material_ref("refresh"),
            on_click=_regenerate_action,
            disabled=status.lower() == "resolved",
        )
        action_cols[3].button(
            "Reject",
            width="stretch",
            icon=material_ref("reject"),
            on_click=_reject_action,
            disabled=status.lower() == "resolved",
        )
        action_cols[4].button(
            "Escalate",
            width="stretch",
            icon=material_ref("escalate"),
            on_click=_escalate_action,
            disabled=status.lower() == "escalated",
        )

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
            queue_choices = ["Keep current", *get_queue_choices()]
            priority_choices = ["Keep current", *get_priority_choices()]

            def _confirm_route() -> None:
                result = confirm_routing(str(selected["ticket_id"]), agent="agent")
                st.toast(result.message)

            router_cols[0].button(
                "Confirm",
                width="stretch",
                icon=material_ref("confirm"),
                on_click=_confirm_route,
                disabled=status.lower() == "assigned",
            )

            selected_queue = router_cols[1].selectbox(
                "Correct queue",
                queue_choices,
                index=0,
                key=f"queue_correction_{selected['ticket_id']}",
                label_visibility="collapsed",
            )
            selected_priority = st.selectbox(
                "Priority",
                priority_choices,
                index=0,
                key=f"priority_correction_{selected['ticket_id']}",
                label_visibility="collapsed",
            )

            def _save_correction() -> None:
                queue_value = str(selected_queue).lower()
                priority_value = str(selected_priority).lower()
                if selected_queue == "Keep current":
                    queue_value = str(selected.get("queue", "general")).lower()
                if selected_priority == "Keep current":
                    priority_value = str(selected.get("priority", "medium")).lower()
                result = correct_routing(
                    str(selected["ticket_id"]),
                    queue=queue_value,
                    priority=priority_value,
                    team=get_queue_team_map().get(queue_value, "Support Desk"),
                    agent="agent",
                )
                _run_action_and_refresh(str(selected["ticket_id"]), lambda: result)

            st.button(
                "Save correction",
                key=f"save_correction_{selected['ticket_id']}",
                type="secondary",
                on_click=_save_correction,
            )

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
