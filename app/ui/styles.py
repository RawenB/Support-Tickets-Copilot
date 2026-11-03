"""Shared design system for the Support Copilot Streamlit UI."""

from __future__ import annotations

import streamlit as st

_STYLES = """
<style>
:root {
    --bg: #17181b;
    --surface: #1e1f23;
    --surface-muted: #232428;
    --surface-hover: #2a2b30;
    --line: #33353b;
    --line-strong: #45474f;
    --text: #e5e4e1;
    --text-strong: #f4f3f0;
    --muted: #9698a1;
    --accent: #7e9cc4;
    --accent-hover: #93afd3;
    --accent-soft: rgba(126, 156, 196, 0.16);
    --radius-lg: 8px;
    --radius-md: 6px;
    --radius-sm: 5px;
}

html, body, .stApp {
    background: var(--bg);
}

div[data-testid="stAppViewContainer"],
div[data-testid="stHeader"],
div[data-testid="block-container"] {
    background: transparent;
}

.block-container {
    padding-top: 0.9rem;
    padding-bottom: 2rem;
    max-width: 1460px;
}

/* ------------------------------------------------------------------ Icons */
.material-icon {
    font-family: "Material Symbols Rounded";
    font-weight: 400;
    font-style: normal;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: "liga";
    -webkit-font-smoothing: antialiased;
    font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
    vertical-align: middle;
}

/* ---------------------------------------------------------------- Top bar */
.topbar {
    position: sticky;
    top: 0;
    z-index: 999;
    margin: 0 0 1.25rem;
    padding: 0.85rem 1.1rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    background: var(--surface);
}

.topbar-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}

.topbar-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-strong);
    line-height: 1.2;
}

.topbar-sub {
    font-size: 0.74rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

.topbar-stats {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0;
}

.stat {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: 0 0.95rem;
    border-left: 1px solid var(--line);
}

.stat:first-child {
    border-left: none;
    padding-left: 0;
}

.stat-label {
    display: flex;
    align-items: center;
    gap: 0.32rem;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
}

.stat-label .material-icon { color: var(--muted); }

.stat-value {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text-strong);
}

.stat-value.is-alert { color: #d79a9a; }

.topbar-stats .tag { margin-left: 0.85rem; }

.tag {
    display: inline-block;
    border-radius: var(--radius-sm);
    background: var(--surface-muted);
    border: 1px solid var(--line);
    color: var(--muted);
    padding: 0.16rem 0.45rem;
    font-size: 0.66rem;
    font-weight: 600;
}

/* --------------------------------------------------------------- Sidebar */
section[data-testid="stSidebar"] {
    background: #191a1e;
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.1rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.2rem 0.1rem 0.9rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 0.4rem;
}

.brand-icon {
    display: grid;
    place-items: center;
    width: 34px;
    height: 34px;
    border-radius: var(--radius-md);
    border: 1px solid var(--line-strong);
    background: var(--surface-muted);
    color: var(--accent);
    flex: 0 0 auto;
}

.brand-name {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-strong);
    line-height: 1.2;
}

.brand-sub {
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}

.sidebar-note {
    border-top: 1px solid var(--line);
    padding: 0.75rem 0.1rem 0;
    color: var(--muted);
    font-size: 0.72rem;
    line-height: 1.5;
}

/* ------------------------------------------------------------- Page header */
.page-header {
    margin: 0.1rem 0 1.1rem;
}

.page-header-row {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}

.page-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: var(--radius-md);
    border: 1px solid var(--line);
    background: var(--surface-muted);
    color: var(--accent);
    flex: 0 0 auto;
    margin-top: 0.05rem;
}

.eyebrow {
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}

.page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-strong);
    letter-spacing: -0.01em;
    line-height: 1.2;
}

.page-subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.35rem;
    max-width: 72ch;
}

.section-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.55rem;
}

.section-title {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-strong);
}

.section-title .material-icon { color: var(--muted); }

.section-caption {
    font-size: 0.7rem;
    color: var(--muted);
}

/* ------------------------------------------------------------------- KPIs */
.kpi-tile {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 0.85rem 0.95rem;
    height: 100%;
}

.kpi-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.4rem;
}

.kpi-label {
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
}

.kpi-icon { color: var(--muted); }

.kpi-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-strong);
    margin-top: 0.3rem;
    line-height: 1.15;
}

.kpi-delta {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

.kpi-tile.tone-accent { border-top: 2px solid var(--accent); }
.kpi-tile.tone-danger { border-top: 2px solid #b55a5a; }
.kpi-tile.tone-warning { border-top: 2px solid #b08a52; }
.kpi-tile.tone-success { border-top: 2px solid #6f9a79; }

/* ----------------------------------------------------------------- Panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1rem;
}

.thread-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 0.95rem 1rem;
}

.soft-card {
    background: var(--surface-muted);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 0.75rem 0.85rem;
}

.source-card {
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--surface-muted);
    padding: 0.65rem 0.8rem;
    margin-bottom: 0.5rem;
}

.source-id {
    font-weight: 700;
    font-size: 0.8rem;
    color: var(--text-strong);
}

.empty-state {
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1rem;
    text-align: center;
    color: var(--muted);
    background: var(--surface);
}

.empty-state .empty-icon {
    color: var(--line-strong);
    margin-bottom: 0.4rem;
}

.empty-state .empty-title {
    color: var(--text-strong);
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.25rem;
}

/* --------------------------------------------------------------- Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    border-radius: var(--radius-sm);
    padding: 0.2rem 0.5rem;
    font-size: 0.68rem;
    font-weight: 650;
    border: 1px solid transparent;
}

.priority-critical {
    background: rgba(181, 90, 90, 0.16);
    color: #e2a5a5;
    border-color: rgba(181, 90, 90, 0.36);
}
.priority-high {
    background: rgba(176, 138, 82, 0.16);
    color: #ddb283;
    border-color: rgba(176, 138, 82, 0.36);
}
.priority-medium {
    background: rgba(170, 158, 90, 0.14);
    color: #d6cb92;
    border-color: rgba(170, 158, 90, 0.32);
}
.priority-low {
    background: rgba(111, 154, 121, 0.15);
    color: #a9c9b1;
    border-color: rgba(111, 154, 121, 0.34);
}

.status-pill {
    display: inline-flex;
    align-items: center;
    border-radius: var(--radius-sm);
    padding: 0.2rem 0.5rem;
    font-size: 0.68rem;
    font-weight: 650;
    background: var(--surface-muted);
    color: var(--muted);
    border: 1px solid var(--line);
}

.status-new {
    background: rgba(110, 148, 194, 0.15);
    color: #aec5e0;
    border-color: rgba(110, 148, 194, 0.34);
}
.status-in_review {
    background: rgba(170, 158, 90, 0.14);
    color: #d6cb92;
    border-color: rgba(170, 158, 90, 0.32);
}
.status-escalated {
    background: rgba(181, 90, 90, 0.16);
    color: #e2a5a5;
    border-color: rgba(181, 90, 90, 0.36);
}
.status-resolved {
    background: rgba(111, 154, 121, 0.15);
    color: #a9c9b1;
    border-color: rgba(111, 154, 121, 0.34);
}
.status-assigned {
    background: var(--surface-muted);
    color: var(--muted);
    border-color: var(--line);
}

.inline-badge {
    display: inline-flex;
    align-items: center;
    border-radius: var(--radius-sm);
    padding: 0.16rem 0.42rem;
    font-size: 0.66rem;
    font-weight: 650;
    background: rgba(170, 158, 90, 0.14);
    color: #d6cb92;
    border: 1px solid rgba(170, 158, 90, 0.32);
}

.demo-pill {
    display: inline-block;
    background: var(--surface-muted);
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    padding: 0.16rem 0.5rem;
    font-size: 0.66rem;
    font-weight: 600;
}

.chip {
    display: inline-flex;
    align-items: center;
    border-radius: var(--radius-sm);
    padding: 0.18rem 0.5rem;
    background: var(--surface-muted);
    border: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.68rem;
    font-weight: 500;
}

/* --------------------------------------------------------------- Tickets */
.ticket-meta {
    font-size: 0.7rem;
    color: var(--muted);
    margin: -0.3rem 0 0.6rem 0.1rem;
    display: flex;
    gap: 0.42rem;
    align-items: center;
    flex-wrap: wrap;
}

.ticket-meta .dot { color: var(--line-strong); }

.ticket-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: var(--text-strong);
    line-height: 1.3;
    margin-bottom: 0.4rem;
}

.meta-row {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 0.8rem;
}

.muted { color: var(--muted); }

/* ---------------------------------------------------------------- Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-strong);
    background: var(--surface);
    color: var(--text);
    font-weight: 600;
    transition: background-color 0.14s ease, border-color 0.14s ease;
    min-height: 2.3rem;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #5a5d66;
    background: var(--surface-hover);
    color: var(--text-strong);
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: var(--accent);
    border-color: var(--accent);
    color: #14181d;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
}

.stButton > button[kind="tertiary"] {
    border-color: transparent;
    background: transparent;
    text-align: left;
    justify-content: flex-start;
    padding-left: 0.55rem;
    padding-right: 0.55rem;
}

.stButton > button[kind="tertiary"]:hover {
    background: var(--surface-hover);
    border-color: transparent;
}

/* ---------------------------------------------------------- Inputs / forms */
div[data-baseweb="input"],
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] {
    border-radius: var(--radius-sm) !important;
    border-color: var(--line-strong) !important;
    background: var(--surface) !important;
}

div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-soft) !important;
}

.stTextArea textarea {
    border-radius: var(--radius-md) !important;
    background: #1a1b1f !important;
    color: var(--text) !important;
    border: 1px solid var(--line-strong) !important;
    line-height: 1.55 !important;
}

.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-soft) !important;
}

div[data-testid="stToggle"] label p {
    font-size: 0.78rem;
    color: var(--muted);
}

/* --------------------------------------------------- Expanders / containers */
div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--surface);
    overflow: hidden;
}

div[data-testid="stExpander"] summary {
    font-weight: 600;
    color: var(--text);
}

div[data-testid="stContainer"] {
    border-color: var(--line);
}

/* -------------------------------------------------------------- Charts/df */
div[data-testid="stMetric"] {
    background: transparent;
}

div[data-testid="stDataFrame"] {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--line);
}

.stAlert {
    border-radius: var(--radius-md);
}

.alert-banner {
    background: rgba(176, 138, 82, 0.12);
    border: 1px solid rgba(176, 138, 82, 0.32);
    border-radius: var(--radius-md);
    padding: 0.7rem 0.85rem;
    font-size: 0.82rem;
    line-height: 1.5;
    color: #dcc09a;
}

hr {
    border-color: var(--line) !important;
}

/* ----------------------------------------------------------- Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: #3a3c43;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #4a4c55; }

/* ---------------------------------------------------------------- Telemetry */
footer { visibility: hidden; }

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .page-title { font-size: 1.3rem; }
    .topbar { position: relative; }
    .stat { padding: 0 0.7rem; }
}
</style>
"""


def inject_styles() -> None:
    """Apply the shared design system used across all product views."""
    st.markdown(_STYLES, unsafe_allow_html=True)
