# modules/quality.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------- loading --------------------------------

def load_quality_ready(path: str | Path = "data_private/quality_ready.parquet") -> pd.DataFrame:
    """
    Load the precomputed Quality table built offline in Jupyter.
    Expected columns include at least:
      ['Ticker','Company','ROIC (TTM)','ROCE (TTM)','ROE (TTM)','ROA (TTM)',
       'Free Cash Flow (TTM)','FCFF (TTM)','FCFE (TTM)','FCF Yield (%)',
       'Gross Profit Margin (TTM)','Net Profit Margin (TTM)','EBIT Margin (TTM)',
       'EBITDA Margin (TTM)','Asset Turnover (TTM)','Debt/Equity (TTM)',
       'Debt Service Coverage (TTM)','Interest Coverage (TTM)',
       'Current Ratio (TTM)','Quick Ratio (TTM)','Altman Z-Score','Piotroski F-Score']
    All rates/margins/RO* are decimals (not %); FCF Yield is already a percent value.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    if "Company" in df.columns:
        df["Company"] = df["Company"].astype(str).str.strip()
    if "Ticker" in df.columns:
        df = df.drop_duplicates("Ticker", keep="first")
    return df

# ------------------------------- core builder ----------------------------

# Base (default) columns shown in the Quality section (cash-flow level items are optional)
BASE_COLS = [
    "Ticker","Company",
    "ROIC (TTM)","ROCE (TTM)","ROE (TTM)","ROA (TTM)",
    "FCF Yield (%)",
    "Gross Profit Margin (TTM)","Net Profit Margin (TTM)",
    "EBIT Margin (TTM)","EBITDA Margin (TTM)",
    "Asset Turnover (TTM)","Debt/Equity (TTM)",
    "Debt Service Coverage (TTM)","Interest Coverage (TTM)",
    "Current Ratio (TTM)","Quick Ratio (TTM)",
    "Altman Z-Score","Piotroski F-Score",
]

# Optional columns users may add in “Columns to show”
OPTIONAL_CF = ["Free Cash Flow (TTM)", "FCFF (TTM)", "FCFE (TTM)"]

@st.cache_data(show_spinner=False, ttl=3600)
def build_quality_table(
    base_view: pd.DataFrame,          # needs ['Ticker','Company']
    quality_df: pd.DataFrame,         # loaded via load_quality_ready()
    meta_map: pd.DataFrame | None = None,   # optional: adds Sector/Industry/MC
) -> pd.DataFrame:
    """
    Filter quality_ready to the current universe, prefer Company from base_view,
    and return BASE_COLS first, then (OPTIONAL_CF and any remaining extras).
    If meta_map is provided, it should contain columns:
      ['Ticker','Sector','Industry','Market Cap (GBP)'] to be available as optional fields.
    """
    if base_view is None or base_view.empty or quality_df is None or quality_df.empty:
        return pd.DataFrame(columns=BASE_COLS)

    uni = base_view[["Ticker","Company"]].dropna(subset=["Ticker"]).copy()
    uni["Ticker"] = uni["Ticker"].astype(str).str.upper().str.strip()
    uni["Company"] = uni["Company"].astype(str).str.strip()
    tickers = uni["Ticker"].tolist()

    core = quality_df[quality_df["Ticker"].isin(tickers)].copy()
    comp_map = uni.set_index("Ticker")["Company"].to_dict()
    core["Company"] = core["Ticker"].map(comp_map).fillna(core.get("Company", np.nan))

    # reorder rows to match selection order
    order_map = {t: i for i, t in enumerate(tickers)}
    core["__order__"] = core["Ticker"].map(order_map)
    core = core.sort_values("__order__").drop(columns="__order__", errors="ignore")

    # place BASE_COLS first, then optional CF, then any extras
    present_base = [c for c in BASE_COLS if c in core.columns]
    extras = [c for c in core.columns if c not in present_base]
    # keep OPTIONAL_CF near the end of base section but ahead of unknown extras
    opt_cf_present = [c for c in OPTIONAL_CF if c in extras]
    other_extras = [c for c in extras if c not in opt_cf_present]
    core = core[present_base + opt_cf_present + other_extras]

    # merge optional meta (Sector/Industry/Market Cap (GBP)) if provided
    if meta_map is not None and not meta_map.empty:
        core = core.merge(
            meta_map[["Ticker","Sector","Industry","Market Cap (GBP)"]],
            on="Ticker", how="left"
        )

    return core.reset_index(drop=True)

# ------------------------------- UI renderer -----------------------------

def _is_percent_col(colname: str) -> bool:
    """
    Percent-display columns: anything with '%' in the name OR
    well-known decimal rates we should render as percentages.
    """
    c = str(colname).lower()
    if "%" in c:  # e.g., FCF Yield (%)
        return True
    # decimal rates → percent
    keys = ("roic", "roce", "roe", "roa", "margin")
    return any(k in c for k in keys)

def _fmt_pct(v):
    if pd.isna(v): return "•"
    try: return f"{float(v)*100:.2f}%"
    except Exception: return "•"

def _fmt_num(v):
    if pd.isna(v): return "•"
    try: return f"{float(v):.2f}"
    except Exception: return "•"

def render_quality_section(
    df: pd.DataFrame,
    style_table_fn,            # pass app.style_table
    pager_fn,                  # pass app.finviz_pager
    export_buttons_fn,         # pass utils.screener_helpers.export_buttons
    key_prefix: str = "qlt_",
):
    if df is None or df.empty:
        st.info("Quality data unavailable for the current filter selection.")
        return

    # default = BASE_COLS (whatever exists); all others available to add
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

    # sort controls
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        sort_col = st.selectbox("Sort by", options=cols_pick, index=0, key=f"{key_prefix}sort_col")
    with c2:
        order = st.selectbox("Order", ["Ascending","Descending"], key=f"{key_prefix}order")
    sort_asc = (order == "Ascending")

    per_page = st.selectbox("Rows per page", [20,50,100,200], index=0, key=f"{key_prefix}pp")

    # reset pagination if controls change
    fp = "||".join([",".join(cols_pick), sort_col, str(sort_asc), str(per_page), str(len(df))])
    if st.session_state.get(f"{key_prefix}fp") != fp:
        st.session_state[f"{key_prefix}page"] = 1
        st.session_state[f"{key_prefix}fp"] = fp

    view = df[cols_pick].copy().sort_values(sort_col, ascending=sort_asc, kind="mergesort")

    # paginate
    total = len(view)
    page = int(st.session_state.get(f"{key_prefix}page", 1))
    start, end = (page - 1) * per_page, (page - 1) * per_page + per_page
    page_df = view.iloc[start:end].copy()
    page_df.insert(0, "No.", range(start + 1, start + 1 + len(page_df)))

    # ---- build display (2 dp; percent rendering; center bullets) ----
    display = page_df.copy()
    pct_cols = [c for c in display.columns if _is_percent_col(c)]

    for c in pct_cols:
        display[c] = display[c].apply(_fmt_pct)

    for c in display.columns:
        if c in ("No.", "Ticker", "Company") or c in pct_cols:
            continue
        if pd.api.types.is_numeric_dtype(display[c]):
            display[c] = display[c].apply(_fmt_num)

    # color green/red for percent-like columns; leave others neutral
    def _color_from_str(v):
        try:
            x = float(str(v).replace("%",""))
            return "color:#22c55e;" if x > 0 else "color:#ef4444;"
        except Exception:
            return ""

    num_cols_all = [c for c in display.columns if c not in ("No.","Ticker","Company")]
    sty = style_table_fn(display)
    sty = sty.set_properties(subset=num_cols_all, **{"text-align":"right"})
    sty = sty.applymap(lambda v: "text-align:center;" if v == "•" else "", subset=num_cols_all)
    if pct_cols:
        sty = sty.applymap(_color_from_str, subset=pct_cols)

    # scrollable dark container
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

    export_buttons_fn(view, filename="quality")
    st.write("")
    new_page, _ = pager_fn(total, per_page, key=f"{key_prefix}page")
    st.session_state[f"{key_prefix}page"] = new_page