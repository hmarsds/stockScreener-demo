# modules/earnings_momentum.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import re
import streamlit as st

# ------------------------------- loading --------------------------------

def load_em_ready(path: str | Path = "data_private/earnings_momentum_ready.parquet") -> pd.DataFrame:
    """
    Load the precomputed Earnings Momentum table built offline in Jupyter.
    Expected columns (at least):
      ['Ticker','Company','EPS TTM YoY','Quarterly EPS YoY','Revenue TTM YoY',
       'EPS Surprise % (last Q)','Revenue Surprise % (last Q)','SUE (EPS)','Beat/Miss Streak', ...]
    Values for *YoY* and *Surprise %* columns are already in PERCENT (not decimals).
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
    # Ensure unique tickers
    if "Ticker" in df.columns:
        df = df.drop_duplicates("Ticker", keep="first")
    return df


# ------------------------------- core builder ----------------------------

# Base columns visible by default in the EM section
BASE_COLS = [
    "Ticker",
    "Company",
    "EPS TTM YoY",
    "Quarterly EPS YoY",
    "Revenue TTM YoY",
    "EPS Surprise % (last Q)",
    "Revenue Surprise % (last Q)",
    "SUE (EPS)",
    "Beat/Miss Streak",
    # NEW (5 metrics)
    "Day-0 Return",
    "5D Post-release Return",
    "20D Post-release Return",
    "Average SUE (last 4Q)",
    "SUE Volatility (last 8Q)",
]

@st.cache_data(show_spinner=False, ttl=3600)
def build_earnings_momentum_table(
    base_view: pd.DataFrame,            # needs ['Ticker','Company']
    em_ready_df: pd.DataFrame,          # loaded via load_em_ready()
) -> pd.DataFrame:
    """
    Filter the precomputed em_ready to the current universe selection.
    Return a table where BASE_COLS are first; all other columns are included
    so users can add them via the 'Columns to show' UI (e.g., raw Current/Prior, flags).
    """
    if base_view is None or base_view.empty or em_ready_df is None or em_ready_df.empty:
        return pd.DataFrame(columns=BASE_COLS)

    # Universe tickers (order-preserving)
    uni = base_view[["Ticker", "Company"]].dropna(subset=["Ticker"]).copy()
    uni["Ticker"] = uni["Ticker"].astype(str).str.upper().str.strip()
    uni["Company"] = uni["Company"].astype(str).str.strip()
    tickers = uni["Ticker"].tolist()

    # Filter em_ready to universe tickers
    core = em_ready_df[em_ready_df["Ticker"].isin(tickers)].copy()

    # Prefer Company names from base_view when available
    comp_map = uni.set_index("Ticker")["Company"].to_dict()
    core["Company"] = core["Ticker"].map(comp_map).fillna(core.get("Company", np.nan))

    # Reorder rows to match the universe order
    order_map = {t: i for i, t in enumerate(tickers)}
    core["__order__"] = core["Ticker"].map(order_map)
    core = core.sort_values("__order__").drop(columns="__order__", errors="ignore")

    # Column ordering: BASE_COLS first (keep what exists), then everything else
    present_base = [c for c in BASE_COLS if c in core.columns]
    remaining = [c for c in core.columns if c not in present_base]
    core = core[present_base + remaining]

    return core.reset_index(drop=True)


# ------------------------------- UI renderer -----------------------------

# helpers to identify formatting buckets
def _is_percent_col(colname: str) -> bool:
    """
    Percent columns: any with 'yoy', 'surprise %', or 'return' in the name.
    (Returns are already in %, per em_ready.)
    """
    c = str(colname).lower()
    return ("yoy" in c) or ("surprise %" in c) or ("return" in c)

def _is_numeric_col(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _fmt_pct(v):
    if pd.isna(v): return "-"
    try: return f"{float(v):.2f}%"
    except Exception: return "-"

def _fmt_num(v):
    if pd.isna(v): return "-"
    try: return f"{float(v):.2f}"
    except Exception: return "-"

def _color_pos_neg(v):
    if pd.isna(v): return ""
    try: return "color:#22c55e;" if float(v) > 0 else "color:#ef4444;"
    except Exception: return ""

# ---- add this helper near the other formatting helpers ----
def _align_bullets(val):
    # center bullet (missing), otherwise right-align numbers/percent strings
    return "text-align: center;" if val == "•" else "text-align: right;"

def render_earnings_momentum_section(
    df: pd.DataFrame,
    style_table_fn,            # pass app.style_table
    pager_fn,                  # pass app.finviz_pager
    export_buttons_fn,         # pass utils.screener_helpers.export_buttons
    key_prefix: str = "em_",
):
    if df is None or df.empty:
        st.info("Earnings momentum data unavailable for the current filter selection.")
        return

    # BASE first, extras optional
    cols_all = df.columns.tolist()
    base_present = [c for c in BASE_COLS if c in cols_all]
    extras = [c for c in cols_all if c not in base_present]

    cols_pick = st.multiselect(
        "Columns to show",
        options=base_present + extras,
        default=base_present,
        key=f"{key_prefix}cols",
    )
    if not cols_pick:
        st.warning("Please choose at least one column to display.")
        return

    # Sort controls
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        sort_col = st.selectbox("Sort by", options=cols_pick, index=0, key=f"{key_prefix}sort_col")
    with c2:
        order = st.selectbox("Order", ["Ascending","Descending"], key=f"{key_prefix}order")
    sort_asc = (order == "Ascending")

    per_page = st.selectbox("Rows per page", [20,50,100,200], index=0, key=f"{key_prefix}pp")

    # Reset pagination when controls change
    fp = "||".join([",".join(cols_pick), sort_col, str(sort_asc), str(per_page), str(len(df))])
    if st.session_state.get(f"{key_prefix}fp") != fp:
        st.session_state[f"{key_prefix}page"] = 1
        st.session_state[f"{key_prefix}fp"] = fp

    view = df[cols_pick].copy().sort_values(sort_col, ascending=sort_asc, kind="mergesort")

    # Pagination
    total = len(view)
    page = int(st.session_state.get(f"{key_prefix}page", 1))
    start = (page - 1) * per_page
    end = start + per_page
    page_df = view.iloc[start:end].copy()
    page_df.insert(0, "No.", range(start + 1, start + 1 + len(page_df)))

    # ---- Build a DISPLAY copy with 2-dp + % strings and centered bullets ----
    display = page_df.copy()

    # Which columns are percentages?
    pct_cols = [c for c in display.columns if _is_percent_col(c)]
    # Extra numeric to color like %:
    color_extra = [c for c in ["SUE (EPS)", "Beat/Miss Streak", "Average SUE (last 4Q)"] if c in display.columns]
    color_cols = list(set(pct_cols + color_extra))

    # Format % columns as strings with two decimals + %; bullet for NaN
    for c in pct_cols:
        display[c] = display[c].apply(lambda v: "•" if pd.isna(v) else f"{float(v):.2f}%")

    # Round other numeric columns to 2 dp; bullet for NaN
    for c in display.columns:
        if c in ("No.", "Ticker", "Company") or c in pct_cols:
            continue
        if pd.api.types.is_numeric_dtype(display[c]):
            display[c] = display[c].apply(lambda v: "•" if pd.isna(v) else f"{float(v):.2f}")

    # Build dark Styler (reuse your theme) + right-align numbers, center bullets
    sty = style_table_fn(display)
    num_cols_all = [c for c in display.columns if c not in ("No.", "Ticker", "Company")]
    sty = sty.set_properties(subset=num_cols_all, **{"text-align": "right"})
    sty = sty.applymap(lambda v: "text-align:center;" if v == "•" else "", subset=num_cols_all)

    # Color green/red on the string values in color_cols (works by parsing float part)
    def _color_from_str(v):
        try:
            # strip % if present
            x = float(str(v).replace("%", ""))
            return "color:#22c55e;" if x > 0 else "color:#ef4444;"
        except Exception:
            return ""
    if color_cols:
        sty = sty.applymap(_color_from_str, subset=color_cols)

    # ---- Render in scrollable dark container (keeps dark theme) ----
    html = sty.hide(axis="index").to_html()
    st.markdown(
        """
        <div style="overflow-x:auto; border:1px solid #1f2937; border-radius:12px;">
          <div style="min-width:960px;">
        """,
        unsafe_allow_html=True,
    )
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Export current (unpaginated would surprise users)
    export_buttons_fn(view, filename="earnings_momentum")

    st.write("")
    new_page, _ = pager_fn(total, per_page, key=f"{key_prefix}page")
    st.session_state[f"{key_prefix}page"] = new_page