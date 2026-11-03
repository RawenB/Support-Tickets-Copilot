"""Icon registry backed by Streamlit's bundled Material Symbols library.

Two helpers are exposed:

- ``material_ref`` returns the ``:material/name:`` shortcode used by native
  Streamlit widgets (``st.Page``, ``st.button``, ``st.expander``, ...).
- ``material_icon`` returns inline markup for custom HTML components so the
  same icon set can be used inside the hand-written panels and cards.
"""

from __future__ import annotations

import html

ICONS: dict[str, str] = {
    # Navigation
    "inbox": "inbox",
    "review": "pending_actions",
    "incidents": "warning",
    "activity": "person",
    "models": "psychology",
    "data": "monitoring",
    "pipeline": "schema",
    # Actions
    "search": "search",
    "send": "send",
    "refresh": "refresh",
    "reject": "block",
    "escalate": "arrow_upward",
    "confirm": "check",
    "spike": "bolt",
    "login": "login",
    # Content
    "brand": "support_agent",
    "sources": "library_books",
    "shield": "shield",
    "history": "history",
    "agents": "groups",
    "focus": "task_alt",
    "queue": "list_alt",
    "open": "inbox",
    "schedule": "schedule",
    "status": "circle",
    "empty": "inventory_2",
    "filter": "filter_list",
    "ticket": "forum",
    "description": "description",
    "trend": "bar_chart",
    "database": "database",
}


def _name(name: str) -> str:
    return ICONS.get(name, name)


def material_ref(name: str) -> str:
    """Return the shortcode accepted by native Streamlit ``icon`` parameters."""
    return f":material/{_name(name)}:"


def material_icon(name: str, *, size: int = 18, extra_class: str = "") -> str:
    """Return inline Material Symbols markup for custom HTML components."""
    glyph = html.escape(_name(name))
    classes = f"material-icon {extra_class}".strip()
    return f'<span class="{classes}" style="font-size:{size}px;">{glyph}</span>'
