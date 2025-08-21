# modules/technicals.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import streamlit as st
from typing import Callable

# ====================== Load helpers ======================

@st.cache_data(show_spinner=False, ttl=3600)
def load_technicals_ready(kind: str) -> pd.DataFrame | None:
    path_map = {
        "MOM":      "data_private/tech_ready_momentum.parquet",
        "MA":       "data_private/tech_ready_MA.parquet",
        "BANDS":    "data_private/tech_ready_bands.parquet",
        "PATTERNS": "data_private/tech_ready_candlestick_patterns.parquet",
    }
    path = path_map.get(str(kind).upper())
    if not path:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None

    df.columns = [str(c).strip() for c in df.columns]
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    if "Company" in df.columns:
        df["Company"] = df["Company"].astype(str).str.strip()
    return df

# ====================== Formatting / styling ======================

GREEN = "#22c55e"
RED   = "#ef4444"
FG    = "#e5e7eb"

def _fmt2(x) -> str:
    if pd.isna(x): return ""
    try: v = float(x)
    except Exception: return str(x)
    return f"{v:.2f}".rstrip("0").rstrip(".")

def _is_percent_col(name: str) -> bool:
    n = str(name).lower().strip()
    return n.endswith("(%)") or n.endswith("%)") or n.endswith("%") or "vs" in n

def _ensure_company_once(core: pd.DataFrame) -> pd.DataFrame:
    df = core.copy()
    if "Company" in df.columns:
        return df
    low = {c.lower(): c for c in df.columns}
    cx, cy = low.get("company_x"), low.get("company_y")
    if cx or cy:
        if cx and cy:
            comp = df[cx].where(df[cx].notna(), df[cy])
            df.insert(1, "Company", comp)
            df = df.drop(columns=[cx, cy], errors="ignore")
        else:
            df = df.rename(columns={(cx or cy): "Company"})
    return df

def _merge_base(base: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
    left  = base[["Ticker","Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")
    right = tech_df.copy()
    right_cols = [c for c in right.columns if c != "Company"]
    core = left.merge(right[right_cols], how="left", on="Ticker")
    if "Company" not in core.columns and "Company" in tech_df.columns:
        core["Company"] = core["Ticker"].map(tech_df.set_index("Ticker")["Company"])
    return _ensure_company_once(core)

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _styler_dark_colored(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Dark table + coloring with !important so it beats global CSS."""
    sty = df.style.set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color","#0f172a"),("color","#f9fafb"),
            ("border-color","#1f2937"),("font-weight","700")
        ]},
        {"selector": "tbody td", "props": [("border-color","#1f2937"),("color", FG)]},
        {"selector": "tbody tr:nth-child(odd) td",  "props": [("background-color","#0f172a")]},
        {"selector": "tbody tr:nth-child(even) td", "props": [("background-color","#111827")]},
    ])

    # numeric formatting
    num_cols = _numeric_cols(df)
    if num_cols:
        sty = sty.format({c: _fmt2 for c in num_cols})

    # numeric/percent coloring
    def color_num(v):
        if pd.isna(v): return ""
        try: vv = float(v)
        except Exception: return ""
        return f"color: {GREEN if vv>0 else RED} !important; font-weight:700 !important;"

    to_color = set(num_cols)
    for c in df.columns:
        if c not in to_color and _is_percent_col(c):
            vc = pd.to_numeric(df[c], errors="coerce")
            if vc.notna().any(): to_color.add(c)
    if to_color:
        sty = sty.applymap(color_num, subset=list(to_color))

    # booleans
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    if bool_cols:
        def color_bool(b):
            if pd.isna(b): return ""
            return (f"color: {GREEN} !important; font-weight:800 !important;"
                    if bool(b) else
                    f"color: {RED} !important; font-weight:800 !important;")
        sty = sty.applymap(color_bool, subset=bool_cols)

    # blue tickers
    if "Ticker" in df.columns:
        sty = sty.set_properties(subset=["Ticker"], **{"color": "#60a5fa !important", "font-weight":"700"})

    try: sty = sty.hide(axis="index")
    except Exception: pass
    return sty

# ====================== Public API ======================

def render_technicals_section(
    base_view: pd.DataFrame,
    style_table_fn: Callable[[pd.DataFrame], pd.io.formats.style.Styler] | None,  # unused
    pager_fn: Callable,
    export_buttons_fn: Callable[[pd.DataFrame, str], None],
    key_prefix: str = "tech_",
):
    # Bigger/white radio
    st.markdown("""
    <style>
      .tech-dataset-heading{ font-size:1.25rem; font-weight:800; color:#ffffff; margin-bottom:.25rem; }
      [data-testid="stRadio"] *{ color:#ffffff !important; }
      [data-testid="stRadio"] label{ font-size:1.05rem !important; font-weight:800 !important; }
      /* st.table dark + horizontal scroll */
      [data-testid="stTable"]{ background:#0f172a !important; }
      [data-testid="stTable"] > div{ overflow-x:auto !important; }
      [data-testid="stTable"] table{ width:max-content !important; min-width:100% !important; }
      [data-testid="stTable"] th, [data-testid="stTable"] td{
        background:#0f172a !important; border-color:#1f2937 !important;
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='tech-dataset-heading'>Dataset</div>", unsafe_allow_html=True)
    choice = st.radio("", ["MOM", "MA", "BANDS", "PATTERNS"], horizontal=True, key=f"{key_prefix}_kind")

    # reset column selection when switching dataset
    prev = st.session_state.get(f"{key_prefix}_kind_prev")
    if prev != choice:
        st.session_state.pop(f"{key_prefix}_cols", None)
        st.session_state[f"{key_prefix}_page"] = 1
        st.session_state[f"{key_prefix}_kind_prev"] = choice

    tech_df = load_technicals_ready(choice)
    if tech_df is None or tech_df.empty:
        st.info(f"No technicals file found for {choice}.")
        return

    # merge
    base = base_view[["Ticker","Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")
    core = _merge_base(base, tech_df)

    # column picker
    default_cols = [c for c in ["Ticker","Company","Close Local Price","Currency","Volume"] if c in core.columns]
    extra_cols   = [c for c in core.columns if c not in default_cols]
    init_cols    = st.session_state.get(f"{key_prefix}_cols") or (default_cols + extra_cols[:12])
    cols_pick = st.multiselect(
        "Columns to show",
        options=(default_cols + extra_cols),
        default=[c for c in init_cols if c in core.columns],
        key=f"{key_prefix}_cols",
    )
    view = core[cols_pick] if cols_pick else core[default_cols + extra_cols]

    # sort + paginate
    sort_opts = view.columns.tolist()
    sort_col  = st.selectbox("Sort by", options=sort_opts,
                             index=(sort_opts.index("Ticker") if "Ticker" in sort_opts else 0),
                             key=f"{key_prefix}_sort_col")
    order     = st.selectbox("Order", ["Ascending","Descending"], index=0, key=f"{key_prefix}_sort_ord")
    sort_asc  = (order == "Ascending")
    per_page  = st.selectbox("Rows per page", [20, 50, 100, 200], index=0, key=f"{key_prefix}_per_page")

    try:
        view_sorted = view.sort_values(sort_col, ascending=sort_asc, kind="mergesort")
    except Exception:
        view_sorted = view

    total = len(view_sorted)
    page  = int(st.session_state.get(f"{key_prefix}_page", 1))
    pages = max(1, math.ceil(total / per_page))
    page  = max(1, min(page, pages))
    start, end = (page - 1) * per_page, (page - 1) * per_page + per_page
    page_df = view_sorted.iloc[start:end].copy().reset_index(drop=True)

    # table (dark + colored + scrollable)
    st.table(_styler_dark_colored(page_df))

    # paginator BELOW
    page, pages = pager_fn(total, per_page, key=f"{key_prefix}_page")

    # export
    export_buttons_fn(view_sorted, filename=f"technicals_{choice.lower()}")
    st.caption(f"Technicals · {choice} | rows={total} | pages={pages} | showing {len(page_df)}")