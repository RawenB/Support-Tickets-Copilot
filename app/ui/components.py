"""Reusable components for the Support Copilot product experience."""

from __future__ import annotations

import html
from typing import Any, Iterable

import streamlit as st

from app.ui.icons import material_icon, material_ref


def _escape(value: Any) -> str:
    return html.escape(str(value))


def priority_badge(priority: str) -> str:
    """Return HTML markup for a priority label."""
    normalized = (priority or "medium").lower()
    label = normalized.title()
    cls = (
        "priority-low"
        if normalized == "low"
        else "priority-medium"
        if normalized == "medium"
        else "priority-high"
        if normalized == "high"
        else "priority-critical"
    )
    return f'<span class="badge {cls}">{_escape(label)}</span>'


def status_pill(status: str) -> str:
    """Return HTML markup for a status badge."""
    normalized = (status or "new").lower().replace(" ", "_")
    cls = (
        "status-new"
        if normalized == "new"
        else "status-in_review"
        if normalized == "in_review"
        else "status-escalated"
        if normalized == "escalated"
        else "status-resolved"
        if normalized == "resolved"
        else "status-assigned"
        if normalized == "assigned"
        else "status-pill"
    )
    label = normalized.replace("_", " ").title()
    return f'<span class="status-pill {cls}">{_escape(label)}</span>'


def page_header(
    title: str,
    subtitle: str | None = None,
    *,
    eyebrow: str | None = None,
    icon: str | None = None,
) -> None:
    """Render the consistent page title block used by every view."""
    icon_html = (
        f'<span class="page-icon">{material_icon(icon, size=20)}</span>' if icon else ""
    )
    text_parts = []
    if eyebrow:
        text_parts.append(f'<div class="eyebrow">{_escape(eyebrow)}</div>')
    text_parts.append(f'<div class="page-title">{_escape(title)}</div>')
    if subtitle:
        text_parts.append(f'<div class="page-subtitle">{_escape(subtitle)}</div>')
    st.markdown(
        '<div class="page-header">'
        f'<div class="page-header-row">{icon_html}<div>{"".join(text_parts)}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def section_title(
    title: str, caption: str | None = None, *, icon: str | None = None
) -> None:
    """Render a compact section heading with an optional caption."""
    caption_html = (
        f'<span class="section-caption">{_escape(caption)}</span>' if caption else ""
    )
    icon_html = material_icon(icon, size=16) if icon else ""
    st.markdown(
        '<div class="section-head">'
        f'<span class="section-title">{icon_html}{_escape(title)}</span>'
        f"{caption_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def kpi_tile(
    label: str,
    value: str | int | float,
    delta: str | None = None,
    *,
    tone: str = "default",
    icon: str | None = None,
) -> None:
    """Render a single KPI tile."""
    tone_class = f" tone-{tone}" if tone != "default" else ""
    icon_html = (
        f'<span class="kpi-icon">{material_icon(icon, size=16)}</span>' if icon else ""
    )
    delta_html = f'<div class="kpi-delta">{_escape(delta)}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi-tile{tone_class}">'
        '<div class="kpi-head">'
        f'<span class="kpi-label">{_escape(label)}</span>'
        f"{icon_html}</div>"
        f'<div class="kpi-value">{_escape(value)}</div>'
        f"{delta_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def kpi_row(items: Iterable[dict[str, Any]]) -> None:
    """Render a responsive row of KPI tiles from dictionaries."""
    item_list = list(items)
    if not item_list:
        return
    columns = st.columns(len(item_list))
    for index, item in enumerate(item_list):
        with columns[index]:
            kpi_tile(
                str(item.get("label", "")),
                item.get("value", "—"),
                item.get("delta"),
                tone=str(item.get("tone", "default")),
                icon=item.get("icon"),
            )


def empty_state(title: str, message: str, *, icon: str = "empty") -> None:
    """Render a calm empty state with a large muted icon."""
    st.markdown(
        '<div class="empty-state">'
        f'<div class="empty-icon">{material_icon(icon, size=28)}</div>'
        f'<div class="empty-title">{_escape(title)}</div>'
        f'<div>{_escape(message)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def confidence_bar(value: float, *, caption: str | None = None) -> None:
    """Render a horizontal confidence bar for routing predictions."""
    safe = max(0.0, min(1.0, float(value)))
    caption_html = (
        f'<div class="muted" style="font-size:0.7rem; margin-top:0.3rem;">'
        f"{_escape(caption)}</div>"
        if caption
        else ""
    )
    st.markdown(
        '<div style="margin-top: 0.35rem; margin-bottom: 0.45rem;">'
        '<div style="height: 8px; border-radius: 999px;'
        ' background: rgba(148, 163, 184, 0.16); overflow: hidden;">'
        f'<div style="width: {safe * 100:.0f}%; height: 100%;'
        ' background: var(--accent); border-radius: 999px;"></div></div>'
        f"{caption_html}</div>",
        unsafe_allow_html=True,
    )


def source_card(source: dict[str, Any]) -> None:
    """Render a retrieved knowledge-base item card."""
    st.markdown(
        '<div class="source-card">'
        '<div style="display:flex; justify-content:space-between; align-items:baseline;">'
        f'<span class="source-id">{_escape(source.get("id", "faq"))}</span>'
        '<span class="muted" style="font-size:0.72rem;">'
        f"score {float(source.get('score', 0.0)):.2f}</span></div>"
        '<div style="font-size:0.83rem; line-height:1.5; margin-top:0.3rem;">'
        f'{_escape(source.get("text", ""))}</div></div>',
        unsafe_allow_html=True,
    )


def alert_banner(message: str, *, tone: str = "warning") -> None:
    """Render a banner to highlight active incidents or review warnings."""
    if tone == "danger":
        color = "#e2a5a5"
        background = "rgba(181, 90, 90, 0.14)"
        border = "rgba(181, 90, 90, 0.36)"
    else:
        color = "#dcc09a"
        background = "rgba(176, 138, 82, 0.12)"
        border = "rgba(176, 138, 82, 0.32)"
    st.markdown(
        f'<div class="alert-banner" style="background:{background};'
        f' border-color:{border}; color:{color};">{_escape(message)}</div>',
        unsafe_allow_html=True,
    )


def ticket_row(
    ticket: dict[str, Any],
    *,
    selected: bool = False,
    key: str | None = None,
) -> bool:
    """Render a clickable ticket row and return whether it was clicked."""
    ticket_id = _escape(ticket.get("ticket_id", "TKT-0"))
    subject = _escape(str(ticket.get("subject", "Untitled ticket")))
    if len(subject) > 46:
        subject = f"{subject[:43]}…"
    label = f"#{ticket_id} — {subject}"
    clicked = st.button(
        label,
        key=key or f"ticket_{ticket_id}",
        width="stretch",
        type="primary" if selected else "tertiary",
        icon=material_ref("ticket"),
    )
    meta_bits = [
        _escape(str(ticket.get("queue", "general")).title()),
        _escape(str(ticket.get("status", "new")).replace("_", " ").title()),
        _escape(ticket.get("age", "recent")),
    ]
    review_html = (
        '<span class="inline-badge">Review</span>'
        if float(ticket.get("confidence", 0.8)) < 0.8
        else ""
    )
    st.markdown(
        '<div class="ticket-meta">'
        + '<span class="dot">·</span>'.join(f"<span>{bit}</span>" for bit in meta_bits)
        + review_html
        + "</div>",
        unsafe_allow_html=True,
    )
    return bool(clicked)
