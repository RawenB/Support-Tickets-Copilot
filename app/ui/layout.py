"""Application chrome: sidebar brand and the live top bar."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from app.ui.icons import material_icon


def _escape(value: Any) -> str:
    return html.escape(str(value))


def sidebar_brand() -> None:
    """Render the product wordmark shown at the top of the sidebar."""
    st.markdown(
        '<div class="brand">'
        f'<span class="brand-icon">{material_icon("brand", size=19)}</span>'
        "<div>"
        '<div class="brand-name">Support Copilot</div>'
        '<div class="brand-sub">Support operations</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def sidebar_note(message: str) -> None:
    """Render a small contextual note at the bottom of the sidebar."""
    st.markdown(f'<div class="sidebar-note">{_escape(message)}</div>', unsafe_allow_html=True)


def _stat(label: str, value: str, *, icon: str | None = None, alert: bool = False) -> str:
    alert_class = " is-alert" if alert else ""
    icon_html = material_icon(icon, size=14) if icon else ""
    return (
        '<div class="stat">'
        f'<span class="stat-label">{icon_html}{_escape(label)}</span>'
        f'<span class="stat-value{alert_class}">{_escape(value)}</span>'
        "</div>"
    )


def render_topbar(
    metrics: dict[str, Any],
    *,
    demo_mode: bool = False,
    title: str = "Support operations",
    subtitle: str = "Live queue, review, and incident health",
) -> None:
    """Render the status bar above every view."""
    stats = [
        _stat("Open", str(metrics.get("open", 0)), icon="inbox"),
        _stat(
            "Needs review",
            str(metrics.get("needs_review", 0)),
            icon="review",
            alert=bool(metrics.get("needs_review")),
        ),
        _stat(
            "Escalated",
            str(metrics.get("escalated", 0)),
            icon="incidents",
            alert=bool(metrics.get("escalated")),
        ),
        _stat(
            "Avg first response",
            f"{metrics.get('avg_first_response_minutes', 8)} min",
            icon="schedule",
        ),
    ]
    demo_tag = '<span class="tag">Demo data</span>' if demo_mode else ""

    st.markdown(
        '<div class="topbar"><div class="topbar-inner">'
        "<div>"
        f'<div class="topbar-title">{_escape(title)}</div>'
        f'<div class="topbar-sub">{_escape(subtitle)}</div>'
        "</div>"
        f'<div class="topbar-stats">{"".join(stats)}{demo_tag}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
