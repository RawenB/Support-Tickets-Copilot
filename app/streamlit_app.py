from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    for candidate in (project_root, src_root):
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from app.services.ticket_service import get_dashboard_metrics, is_demo_mode
from app.ui.icons import material_ref
from app.ui.layout import render_topbar, sidebar_brand, sidebar_note
from app.ui.styles import inject_styles
from app.views.activity import activity_view
from app.views.admin_data import admin_data_view
from app.views.admin_models import admin_models_view
from app.views.admin_pipeline import admin_pipeline_view
from app.views.inbox import inbox_view
from app.views.incidents import incidents_view
from app.views.needs_review import needs_review_view

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Support Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

st.session_state.setdefault("authenticated", True)

if not st.session_state["authenticated"]:
    st.markdown(
        '<div style="max-width:420px; margin:12vh auto 0;">'
        '<div class="page-title">Support Copilot</div>'
        '<div class="page-subtitle">Sign in to continue to the support workspace.</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.form("login", border=True):
        st.text_input("Agent email", key="agent_email")
        st.text_input("Password", type="password")
        if st.form_submit_button(
            "Continue", type="primary", width="stretch", icon=material_ref("login")
        ):
            st.session_state["authenticated"] = True
            st.rerun()
    st.stop()

with st.sidebar:
    sidebar_brand()
    admin_mode = st.toggle(
        "Admin mode",
        value=False,
        help="Enable admin-only operational views for models, data, and pipelines.",
    )
    sidebar_note(
        "Human-in-the-loop by design. The agent confirms every outbound reply."
    )

workspace_pages = [
    st.Page(inbox_view, title="Inbox", icon=material_ref("inbox"), url_path="inbox"),
    st.Page(
        needs_review_view,
        title="Needs review",
        icon=material_ref("review"),
        url_path="review",
    ),
    st.Page(
        incidents_view,
        title="Incidents",
        icon=material_ref("incidents"),
        url_path="incidents",
    ),
    st.Page(
        activity_view,
        title="My activity",
        icon=material_ref("activity"),
        url_path="activity",
    ),
]

navigation: dict[str, list[st.Page]] = {"Workspace": workspace_pages}

if admin_mode:
    navigation["Administration"] = [
        st.Page(
            admin_models_view,
            title="Models & versions",
            icon=material_ref("models"),
            url_path="admin-models",
        ),
        st.Page(
            admin_data_view,
            title="Data & drift",
            icon=material_ref("data"),
            url_path="admin-data",
        ),
        st.Page(
            admin_pipeline_view,
            title="Pipeline",
            icon=material_ref("pipeline"),
            url_path="admin-pipeline",
        ),
    ]

main = st.navigation(navigation, position="sidebar")

render_topbar(get_dashboard_metrics(), demo_mode=is_demo_mode())

main.run()
