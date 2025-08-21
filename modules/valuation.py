# modules/valuation.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------- loading --------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_valuation_ready(path: str | Path = "data_private/valuation_ready.parquet") -> pd.DataFrame:
    """
    Load the precomputed Valuation table built offline in Jupyter.
    Expected columns include at least:
      ['Ticker','Company',
       'P/E (TTM)','PEG (TTM)','Forward PEG','P/B (TTM)','P/S (TTM)',
       'P/FCF (TTM)','EV/Sales (TTM)','EV/EBITDA (TTM)','Dividend Yield (%)']
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)

    # Normalize minimal fields
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    if "Company" in df.columns:
        df["Company"] = df["Company"].astype(str).str.strip()

    # ensure unique tickers
    if "Ticker" in df.columns:
        df = df.drop_duplicates("Ticker", keep="first")

    return df


# ------------------------------- optional profile cols -------------------

def _humanize_mc(x):
    if x is None or pd.isna(x): return "-"
    v = float(x)
    if v >= 1e12: return f"{v/1e12:.2f}T"
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    if v >= 1e6:  return f"{v/1e6:.2f}M"
    if v >= 1e3:  return f"{v/1e3:.0f}K"
    return f"{v:.0f}"

def add_optional_profile_cols(df: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Sector / Industry / Market Cap (GBP) so users can include them
    via the 'Columns to show' control. Not selected by default.
    """
    if df is None or df.empty or universe_df is None or universe_df.empty:
        return df

    u = universe_df.copy()
    u["ticker"] = u["ticker"].astype(str).str.upper().str.strip()
    sector_map   = u.set_index("ticker")["sector"].to_dict()
    industry_map = u.set_index("ticker")["industry"].to_dict()
    mcap_map     = u.set_index("ticker")["market_cap_gbp"].to_dict()

    if "Ticker" in df.columns:
        df["Sector"]   = df["Ticker"].map(sector_map)
        df["Industry"] = df["Ticker"].map(industry_map)
        df["Market Cap (GBP)"] = df["Ticker"].map(mcap_map).apply(_humanize_mc)

    return df


# ------------------------------- core builder ----------------------------

VAL_BASE_COLS = [
    "Ticker","Company",
    "P/E (TTM)","PEG (TTM)","Forward PEG",
    "P/B (TTM)","P/S (TTM)","P/FCF (TTM)",
    "EV/Sales (TTM)","EV/EBITDA (TTM)","Dividend Yield (%)",
]

@st.cache_data(show_spinner=False, ttl=3600)
def build_valuation_table(
    base_view: pd.DataFrame,            # requires ['Ticker','Company']
    valuation_df: pd.DataFrame,         # loaded via load_valuation_ready()
) -> pd.DataFrame:
    """
    Filter valuation_df to the *current* universe selection (preserve order),
    and put the section's base columns first.
    """
    if base_view is None or base_view.empty or valuation_df is None or valuation_df.empty:
        return pd.DataFrame(columns=VAL_BASE_COLS)

    uni = base_view[["Ticker","Company"]].dropna(subset=["Ticker"]).copy()
    uni["Ticker"]  = uni["Ticker"].astype(str).str.upper().str.strip()
    uni["Company"] = uni["Company"].astype(str).str.strip()
    tickers = uni["Ticker"].tolist()

    core = valuation_df[valuation_df["Ticker"].isin(tickers)].copy()

    # Prefer Company from the current universe when present
    comp_map = uni.set_index("Ticker")["Company"].to_dict()
    core["Company"] = core["Ticker"].map(comp_map).fillna(core.get("Company", np.nan))

    # Reorder rows to match selection order
    order_map = {t: i for i, t in enumerate(tickers)}
    core["__order__"] = core["Ticker"].map(order_map)
    core = core.sort_values("__order__").drop(columns="__order__", errors="ignore")

    # Base columns first, keep any extras at the end
    present_base = [c for c in VAL_BASE_COLS if c in core.columns]
    remaining = [c for c in core.columns if c not in present_base]
    return core[present_base + remaining].reset_index(drop=True)


# ------------------------------- UI renderer -----------------------------

def _is_percent_col(colname: str) -> bool:
    return "%" in str(colname)

def _fmt_pct_str(v):
    if pd.isna(v): return "•"
    try: return f"{float(v):.2f}%"
    except Exception: return "•"

def _fmt_num_str(v):
    if pd.isna(v): return "•"
    try: return f"{float(v):.2f}"
    except Exception: return "•"

def render_valuation_section(
    df: pd.DataFrame,
    style_table_fn,            # pass app.style_table (dark theme Styler)
    pager_fn,                  # pass app.finviz_pager
    export_buttons_fn,         # pass utils.screener_helpers.export_buttons
    key_prefix: str = "val_",
):
    if df is None or df.empty:
        st.info("Valuation data unavailable for the current filter selection.")
        return

    # Default = base columns; extras (incl. Sector/Industry/Market Cap) are optional
    cols_all = df.columns.tolist()
    base_present = [c for c in VAL_BASE_COLS if c in cols_all]
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

    # Paginate
    total = len(view)
    page = int(st.session_state.get(f"{key_prefix}page", 1))
    start = (page - 1) * per_page
    end = start + per_page
    page_df = view.iloc[start:end].copy()
    page_df.insert(0, "No.", range(start + 1, start + 1 + len(page_df)))

    # Build DISPLAY copy (2 d.p., '%' symbol for percent columns, bullets for NaN)
    display = page_df.copy()
    pct_cols = [c for c in display.columns if _is_percent_col(c)]

    for c in pct_cols:
        display[c] = display[c].apply(_fmt_pct_str)

    for c in display.columns:
        if c in ("No.","Ticker","Company") or c in pct_cols:
            continue
        if pd.api.types.is_numeric_dtype(display[c]):
            display[c] = display[c].apply(_fmt_num_str)

    # Dark Styler + alignment (numbers right, bullets centered)
    sty = style_table_fn(display)
    num_cols_all = [c for c in display.columns if c not in ("No.","Ticker","Company")]
    sty = sty.set_properties(subset=num_cols_all, **{"text-align":"right"})
    sty = sty.applymap(lambda v: "text-align:center;" if v == "•" else "", subset=num_cols_all)

    # Render inside a horizontal scroll container
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

    # Export current page’s view (consistent with other sections)
    export_buttons_fn(view, filename="valuation")

    st.write("")
    new_page, _ = pager_fn(total, per_page, key=f"{key_prefix}page")
    st.session_state[f"{key_prefix}page"] = new_page