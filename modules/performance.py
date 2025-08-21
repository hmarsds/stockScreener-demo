# modules/performance.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ---------- Loading (precomputed performance) ----------
def load_performance_ready(path: str | Path = "data_private/performance_ready.parquet") -> pd.DataFrame:
    """
    Load precomputed performance metrics (decimal form) with columns:
    ['Ticker','Company','Past 7 days %','1M %','3M %','6M %','YTD %','1Y %','3Y %',
     '% from 52W High','% from 52W Low','1Y CAGR','3Y CAGR'].
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)

    # Normalize
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    if "Company" in df.columns:
        df["Company"] = df["Company"].astype(str).str.strip()

    # Ensure numeric columns are numeric (in decimal form)
    metric_cols = [c for c in df.columns if c not in ("Ticker", "Company")]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------- UI rendering ----------
def render_performance_section(
    perf_df: pd.DataFrame,
    style_table_fn: Optional[Callable[[pd.DataFrame], "pd.io.formats.style.Styler"]] = None,
    pager_fn: Optional[Callable[[int, int, str], tuple[int, int]]] = None,
    export_buttons_fn: Optional[Callable[[pd.DataFrame, str], None]] = None,
) -> None:
    """
    Renders the Performance table using precomputed metrics (decimal form).
    - style_table_fn: optional function(df)->Styler to reuse your Overview styler.
    - pager_fn: optional finviz-style pager(total_rows, per_page, key) -> (page, pages).
    - export_buttons_fn: optional export function(df, filename).
    """
    if perf_df is None or perf_df.empty:
        st.info("Performance data unavailable for the current filter selection.")
        return

    # Column chooser
    all_cols = [
        "Ticker","Company","Past 7 days %","1M %","3M %","6M %",
        "YTD %","1Y %","3Y %","% from 52W High","% from 52W Low","1Y CAGR","3Y CAGR"
    ]
    present_cols = [c for c in all_cols if c in perf_df.columns]
    cols_pick = st.multiselect("Columns to show", options=present_cols, default=present_cols, key="perf_cols")

    # Sort + order
    sort_candidates = [c for c in cols_pick if c != "Company"]
    if "Ticker" not in sort_candidates and "Ticker" in cols_pick:
        sort_candidates = ["Ticker"] + sort_candidates

    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        sort_col = st.selectbox("Sort by", options=sort_candidates,
                                index=sort_candidates.index("Ticker") if "Ticker" in sort_candidates else 0,
                                key="perf_sort_col")
    with c2:
        order = st.selectbox("Order", ["Ascending","Descending"], key="perf_sort_order")
    sort_asc = (order == "Ascending")

    per_page = st.selectbox("Rows per page", [20, 50, 100, 200], index=0, key="perf_per_page")

    # Reset pagination when controls change
    key_parts = (",".join(cols_pick), sort_col, str(sort_asc), str(per_page), str(len(perf_df)))
    fp = "||".join(key_parts)
    if st.session_state.get("perf_fingerprint") != fp:
        st.session_state["perf_page"] = 1
        st.session_state["perf_fingerprint"] = fp

    # Sort entire dataset
    to_show = perf_df[cols_pick].copy()
    if sort_col in to_show.columns:
        to_show = to_show.sort_values(sort_col, ascending=sort_asc, kind="mergesort")

    # Pagination
    page = int(st.session_state.get("perf_page", 1))
    total = len(to_show)
    pages = max(1, math.ceil(total / per_page))
    start = (page - 1) * per_page
    end = start + per_page
    page_df = to_show.iloc[start:end].copy()
    page_df.insert(0, "No.", range(start + 1, start + 1 + len(page_df)))

    # Formatting (percent strings + green/red) and style (match Overview)
    pct_cols = [c for c in page_df.columns if c not in ("No.","Ticker","Company")]

    def _fmt_pct(v):
        if pd.isna(v): return "-"
        try: return f"{float(v)*100:.2f}%"
        except Exception: return "-"

    def _color_pct(v):
        if pd.isna(v): return ""
        try: return "color:#22c55e;" if float(v) > 0 else "color:#ef4444;"
        except Exception: return ""

    def _local_zebra(sty):
        return sty.set_table_styles([
            {"selector": "thead th",
             "props": [("background-color","#0f172a"),
                       ("color","#f9fafb"),
                       ("border-color","#1f2937")]},
            {"selector": "tbody td",
             "props": [("border-color","#1f2937")]},
            {"selector": "tbody tr:nth-child(odd) td",
             "props": [("background-color","#0f172a")]},
            {"selector": "tbody tr:nth-child(even) td",
             "props": [("background-color","#111827")]},
        ]).set_properties(**{"color":"#e5e7eb","border-color":"#1f2937"})

    if style_table_fn is not None:
        sty = style_table_fn(page_df)
    else:
        sty = _local_zebra(page_df.style)

    sty = sty.format({c: _fmt_pct for c in pct_cols}).applymap(_color_pct, subset=pct_cols)

    # Render as a static styled table (like Overview)
    try:
        st.table(sty.hide(axis="index"))
    except Exception:
        st.table(sty)

    # Export CSV
    if export_buttons_fn is not None:
        export_buttons_fn(to_show, filename="performance")
    else:
        csv = to_show.to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", data=csv, file_name="performance.csv", mime="text/csv", key="perf_export")

    # Pager under the table
    st.write("")
    if pager_fn is not None:
        new_page, _ = pager_fn(total, per_page, key="perf_page")
        st.session_state["perf_page"] = new_page
    else:
        c_prev, c_page, c_next = st.columns([0.1, 0.8, 0.1])
        with c_prev:
            if st.button("◀", key="perf_prev", use_container_width=True, disabled=(page <= 1)):
                page = max(1, page - 1)
        with c_page:
            st.markdown(f"<div style='text-align:center; opacity:.85;'>Page {page} / {pages}</div>", unsafe_allow_html=True)
        with c_next:
            if st.button("▶", key="perf_next", use_container_width=True, disabled=(page >= pages)):
                page = min(pages, page + 1)
        st.session_state["perf_page"] = page