import streamlit as st

_BASE_DARK = """
:root { color-scheme: dark; }
html, body, .stApp { background:#0b1020 !important; color:#f9fafb !important; }

/* ---------- Spinner (st.spinner) → dark, no white box ---------- */
[data-testid="stSpinner"]{
  background:#0f172a !important;         /* dark box */
  color:#f9fafb !important;               /* white text */
  border:1px solid #233046 !important;    /* subtle border */
  box-shadow:none !important;             /* kill white drop-shadow */
}
[data-testid="stSpinner"] *{
  color:#f9fafb !important;
}
[data-testid="stSpinner"] svg{
  stroke:#22c55e !important;              /* green ring */
}

/* Push app down so spinner + deploy bar never overlap */
[data-testid="stHeader"]{
  background:#0b1020 !important; color:#f9fafb !important;
  box-shadow:none !important; border-bottom:1px solid #111827 !important;
}
.block-container { padding-top:3.6rem !important; padding-bottom:1.25rem !important; }

/* ---------- Filter pills (active = bright green) ---------- */
.pill-group .stButton>button{
  padding:.70rem 1.2rem;border-radius:28px;border:1px solid #233046;
  background:#0d1427;color:#ffffff;font-weight:800;
}
/* Streamlit marks active buttons with kind="primary" */
.pill-group .stButton>button[kind="primary"],
.pill-group .stButton>button[data-testid="baseButton-primary"]{
  border-color:#22c55e !important;
  box-shadow:inset 0 0 0 2px #22c55e !important;
  background:#0d1a2f !important;
  color:#22c55e !important;
}

/* ---------- Data tabs (active = bright green) ---------- */
.tab-group .stButton>button{
  width:100%; padding:.70rem 1.0rem;border-radius:18px;border:1px solid #233046;
  background:#0d1427;color:#ffffff;font-weight:800;
}
.tab-group .stButton>button[kind="primary"],
.tab-group .stButton>button[data-testid="baseButton-primary"]{
  border-color:#22c55e !important;
  box-shadow:inset 0 0 0 2px #22c55e !important;
  background:#0f172a !important;
  color:#22c55e !important;
}

/* ---------- DataFrame ---------- */
div[data-testid="stDataFrame"] { background:#0f172a !important; }
div[data-testid="stDataFrame"] table { color:#f9fafb !important; }
div[data-testid="stDataFrame"] th, 
div[data-testid="stDataFrame"] td { background:#0f172a !important; border-color:#1f2937 !important; }

/* ---------- Slider (compact + white labels) ---------- */
.stSlider [data-baseweb="slider"]{ height:12px; }
.stSlider [role="slider"]{ box-shadow:none !important; }
.stSlider, .stSlider * { color:#f9fafb !important; }
/* Hide the tick/labels row under the slider – keep only the values above */
.stSlider [data-testid="stTickBar"],
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { display:none !important; }

/* Extra label under MC slider */
.mc-label{ margin-top:.35rem; font-size:.85rem; color:#f9fafb; }

/* ---------- Multiselect (darker + brighter blue chips, white text) ---------- */
.stMultiSelect [data-baseweb="select"]>div{ background:#0f172a !important; color:#f9fafb !important; }
.stMultiSelect [data-baseweb="tag"]{
  background:#1c2b4a !important; color:#ffffff !important; border:1px solid #2a3d66 !important;
}
.stMultiSelect svg{ color:#cbd5e1 !important; }

/* All / None small buttons BELOW each multiselect */
.chip-row{ display:flex; gap:.5rem; margin:.35rem 0 .6rem 0; }
.chip-row .stButton>button{
  padding:.30rem .60rem; font-size:.80rem; line-height:1.1;
  border-radius:10px; border:1px solid #334155; background:#0d1427; color:#f9fafb;
}
.chips-below .stButton>button{
  padding:.28rem .7rem; font-size:.78rem; line-height:1; 
  border-radius:12px; border:1px solid #334155; background:#0d1427; color:#ffffff;
}

/* Pagination below table */
.pager { display:flex; gap:.4rem; align-items:center; justify-content:center; margin-top:.75rem; }
.pager .stButton>button{
  background:#0d1427; color:#f9fafb; border:1px solid #233046; 
  padding:.35rem .7rem; border-radius:9999px;
}
.pager .stButton>button[data-testid="baseButton-primary"]{
  border-color:#22c55e !important; box-shadow:inset 0 0 0 2px #22c55e !important; background:#0f172a !important; color:#22c55e !important;
}

/* Header icon sizing */
.page-header-icon {font-size:40px; line-height:1;}


/* Hide the duplicate min/max labels rendered beneath the slider */
.stSlider [data-baseweb="slider"] > div:last-child { display:none !important; }
/* Back-compat selector on older Streamlit builds */
.stSlider [data-testid="stTickBar"],
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { display:none !important; }

/* Make the All/None buttons compact while keeping text on one line */
button[kind="secondary"] { white-space: nowrap; }

/* Make widget labels (e.g., Sector / Industry / Country) pure white */
.stSlider label,
.stMultiSelect label,
.stTextInput label,
.stSelectbox label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] > div { color:#ffffff !important; }

/* Hide the duplicate min/max tickbar UNDER the slider (keep only the values above) */
.stSlider [data-baseweb="slider"] + div { display:none !important; }
.stSlider [data-testid="stTickBar"],
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { display:none !important; }

/* Global: active Streamlit button (used by our pills & tabs via type="primary") */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"]{
  border-color:#22c55e !important;
  box-shadow: inset 0 0 0 2px #22c55e !important;
  background:#0f172a !important;
  color:#22c55e !important;
}

/* Base look for our pills/tabs (secondary state) */
.stButton > button[kind="secondary"]{
  border:1px solid #233046 !important;
  background:#0d1427 !important;
  color:#ffffff !important;
  font-weight:800;
  border-radius:28px;   /* pills */
}

/* --- Select & Multiselect inputs: dark field + dark dropdown menu --- */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div{
  background:#0f172a !important;
  color:#f9fafb !important;
  border-color:#233046 !important;
}
.stSelectbox svg, .stMultiSelect svg{ color:#cbd5e1 !important; }

/* Dropdown menus (BaseWeb) */
.stSelectbox [data-baseweb="popover"] [data-baseweb="menu"],
.stMultiSelect [data-baseweb="popover"] [data-baseweb="menu"]{
  background:#0f172a !important;
  color:#f9fafb !important;
  border:1px solid #233046 !important;
}
.stSelectbox [data-baseweb="menu"] [role="option"],
.stMultiSelect [data-baseweb="menu"] [role="option"]{
  color:#f9fafb !important;
}
.stSelectbox [data-baseweb="menu"] [role="option"]:hover,
.stMultiSelect [data-baseweb="menu"] [role="option"]:hover{
  background:#172036 !important;
}

/* --- Streamlit DataFrame (grid) hard-dark + white text + zebra --- */
[data-testid="stDataFrame"] * { color:#f1f5f9 !important; }

[data-testid="stDataFrame"] div[role="grid"]{
  background:#0f172a !important;
  border-color:#1f2937 !important;
}

/* Header row */
[data-testid="stDataFrame"] div[role="columnheader"]{
  background:#0f172a !important;
  color:#f9fafb !important;
  border-color:#1f2937 !important;
}

/* Body rows (zebra) */
[data-testid="stDataFrame"] div[role="rowgroup"] > div[role="row"]{
  background:#0f172a !important;
  border-color:#1f2937 !important;
}
[data-testid="stDataFrame"] div[role="rowgroup"] > div[role="row"]:nth-child(even){
  background:#111827 !important;
}

/* --- Download / Export button: cyan border + dark background --- */
.stDownloadButton > button{
  background:#0d1427 !important;
  border:1px solid #06b6d4 !important;   /* cyan border */
  color:#67e8f9 !important;              /* cyan text */
  border-radius:12px !important;
  font-weight:700 !important;
}
.stDownloadButton > button:hover{
  background:#0f172a !important;
  box-shadow:inset 0 0 0 2px #06b6d4 !important;
  color:#a5f3fc !important;              /* brighter cyan on hover */
}

/* === Select & Multiselect — force bright text in the control AND the menu === */

/* Closed control: field + value text + caret */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div{
  background:#0f172a !important;
  border-color:#233046 !important;
}
.stSelectbox [data-baseweb="select"] *,
.stMultiSelect [data-baseweb="select"] *{
  color:#ffffff !important;            /* make all inner text white */
}
.stSelectbox svg, .stMultiSelect svg{ color:#cbd5e1 !important; }

/* Chips in multiselect */
.stMultiSelect [data-baseweb="tag"]{
  background:#1c2b4a !important;
  color:#ffffff !important;
  border:1px solid #2a3d66 !important;
}

/* Open dropdown menu */
.stSelectbox [data-baseweb="popover"] [data-baseweb="menu"],
.stMultiSelect [data-baseweb="popover"] [data-baseweb="menu"]{
  background:#0f172a !important;
  color:#ffffff !important;
  border:1px solid #233046 !important;
}

/* Options inside the menu */
.stSelectbox [data-baseweb="menu"] [role="option"],
.stMultiSelect [data-baseweb="menu"] [role="option"]{
  color:#ffffff !important;
}
.stSelectbox [data-baseweb="menu"] [role="option"]:hover,
.stMultiSelect [data-baseweb="menu"] [role="option"]:hover{
  background:#172036 !important;
  color:#ffffff !important;
}
.stSelectbox [data-baseweb="menu"] [role="option"][aria-selected="true"],
.stMultiSelect [data-baseweb="menu"] [role="option"][aria-selected="true"]{
  background:#1f2a44 !important;
  color:#ffffff !important;
}

/* --- Text input (st.text_input) : dark field + white text --- */
.stTextInput [data-baseweb="input"] > div {
  background:#0f172a !important;       /* field background */
  border:1px solid #233046 !important;  /* subtle border */
  color:#ffffff !important;             /* text color */
}
.stTextInput input {
  background:#0f172a !important;
  color:#ffffff !important;
  caret-color:#ffffff !important;
}
.stTextInput input::placeholder {
  color:#cbd5e1 !important;             /* placeholder readable on dark */
  opacity:1 !important;
}
/* Focus state */
.stTextInput [data-baseweb="input"] > div:focus-within {
  border-color:#3b82f6 !important;      /* thin blue focus ring */
  box-shadow:inset 0 0 0 1px #3b82f6 !important;
}

"""

def inject_dark_theme():
    try:
        with open("assets/dark.css", "r") as f:
            css = f.read()
    except FileNotFoundError:
        css = ""
    st.markdown(f"<style>{css}\n{_BASE_DARK}</style>", unsafe_allow_html=True)

def header(icon: str, title: str, subtitle: str = ""):
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.08, 0.92])
    with c1: st.markdown(f"<div class='page-header-icon'>{icon}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"### {title}")
        if subtitle: st.caption(subtitle)
    st.markdown("---")