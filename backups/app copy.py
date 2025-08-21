# app.py — L/S Stock Screener (Overview + Earnings Momentum)
import math
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from utils.theme import inject_dark_theme, header
from utils.data import load_all
from utils.screener_helpers import export_buttons
from modules.performance import (
    load_performance_ready,
    render_performance_section,
)
from modules.earnings_momentum import (
    load_em_ready,
    build_earnings_momentum_table,
    render_earnings_momentum_section,
)


try:
    from pandas.io.formats.style import Styler as PandasStyler  # type: ignore
except Exception:
    PandasStyler = None  # type: ignore

# ---------------------------- Page / Theme ----------------------------
st.set_page_config(page_title="L/S Stock Screener", page_icon="🕵️‍♂️", layout="wide")
inject_dark_theme()

# ---------------------------- Helpers ----------------------------
COUNTRY_NAME = {
    "US":"USA","GB":"United Kingdom","DE":"Germany","FR":"France","NL":"Netherlands",
    "BE":"Belgium","BB":"Belgium","BM":"Bermuda","BS":"Bahamas","BW":"Botswana",
    "IT":"Italy","ES":"Spain","PT":"Portugal","IE":"Ireland",
    "CH":"Switzerland","AT":"Austria","SE":"Sweden","NO":"Norway","DK":"Denmark",
    "FI":"Finland","CA":"Canada","AU":"Australia","NZ":"New Zealand","JP":"Japan",
    "HK":"Hong Kong","SG":"Singapore","CN":"China","IN":"India","BR":"Brazil",
    "MX":"Mexico","AR":"Argentina","CL":"Chile","CO":"Colombia","ZA":"South Africa",
    "AE":"United Arab Emirates","IL":"Israel","RU":"Russia","CZ":"Czechia","PL":"Poland",
    "HU":"Hungary","IS":"Iceland","TR":"Turkey",
    "CI":"Cote d'Ivoire","CR":"Costa Rica","CW":"Curacao","CY":"Cyprus",
    "EE":"Estonia","EG":"Egypt","GA":"Gabon","GE":"Georgia","GG":"Guernsey",
    "GI":"Gibraltar","GR":"Greece","JE":"Jersey","KR":"South Korea",
    "KY":"Cayman Islands","KZ":"Kazakhstan","LI":"Liechtenstein","LU":"Luxembourg",
    "LT":"Lithuania","MN":"Mongolia","MO":"Macau","MT":"Malta","MU":"Mauritius",
    "MY":"Malaysia","NG":"Nigeria","PE":"Peru","PH":"Philippines","RE":"Reunion",
    "SA":"Saudi Arabia","SI":"Slovenia","TH":"Thailand","TW":"Taiwan",
    "TZ":"Tanzania","UA":"Ukraine","UY":"Uruguay","VG":"British Virgin Islands",
    "VN":"Vietnam",
}
REGION_HINTS = {
    "L":"United Kingdom","OL":"Norway","BA":"Argentina","VI":"Austria","WA":"Poland",
    "DE":"Germany","F":"France","PA":"France","AS":"Netherlands","BR":"Belgium",
    "MC":"Monaco","JO":"South Africa","ST":"Sweden","CO":"Denmark","MI":"Italy",
    "LS":"Portugal","DU":"Germany","AQ":"United Kingdom","BD":"Hungary","IC":"Iceland",
    "MX":"Mexico","IS":"Turkey","TO":"Canada","V":"Canada","NE":"Canada","CN":"Canada",
    "CL":"Colombia","SN":"Chile","ME":"Russia","PR":"Czechia","HM":"France","HE":"Finland",
    "SG":"France","IL":"Israel","TA":"Israel",
}
def country_to_name(code: str) -> str:
    if pd.isna(code) or not str(code).strip(): return "-"
    c = str(code).upper().strip()
    return COUNTRY_NAME.get(c, REGION_HINTS.get(c, c))

def humanize_mc(x):
    if x is None or pd.isna(x): return "-"
    v = float(x)
    if v >= 1e12: return f"{v/1e12:.2f}T"
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    if v >= 1e6:  return f"{v/1e6:.2f}M"
    if v >= 1e3:  return f"{v/1e3:.0f}K"
    return f"{v:.0f}"

def latest_non_null_map_from_wide(wide_df: pd.DataFrame) -> dict:
    if wide_df is None or wide_df.empty: return {}
    df = wide_df.sort_index().ffill()
    last = df.iloc[-1]
    last.index = last.index.astype(str).str.upper().str.strip()
    return last.to_dict()

def build_universe(company_df: pd.DataFrame, mcap_gbp_wide: pd.DataFrame | None) -> pd.DataFrame:
    if company_df is None or company_df.empty:
        return pd.DataFrame(columns=[
            "ticker","name","sector","industry","country","currency",
            "isin","exchangeFullName","market_cap_gbp"
        ])

    df = company_df.rename(columns={"symbol":"ticker","companyName":"name"}).copy()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # ensure needed columns exist
    for c in ("ticker","name","sector","industry","country","currency","isin","exchangeFullName"):
        if c not in df.columns:
            df[c] = pd.NA

    # normalize types/whitespace
    if isinstance(df["ticker"], pd.DataFrame):
        df["ticker"] = df["ticker"].iloc[:,0]
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"]   = df["name"].astype(str).str.strip()
    for c in ("sector","industry","currency","isin","exchangeFullName","country"):
        df[c] = df[c].astype(str).str.strip()

    df = df[[
        "ticker","name","sector","industry","country","currency","isin","exchangeFullName"
    ]].dropna(subset=["ticker"]).drop_duplicates("ticker")

    # latest market cap map (GBP)
    if mcap_gbp_wide is not None and not mcap_gbp_wide.empty:
        df["market_cap_gbp"] = df["ticker"].map(latest_non_null_map_from_wide(mcap_gbp_wide))
    else:
        df["market_cap_gbp"] = np.nan

    # country pretty name
    df["country"] = df["country"].map(country_to_name)
    return df

def style_table(df: pd.DataFrame):
    # Base text & borders
    sty = df.style.set_properties(**{
        "color":"#e5e7eb",
        "border-color":"#1f2937"
    })

    def highlight_placeholders(val):
        if val == "•":
            return "text-align: center;"
        return "text-align: right;"

    # Zebra rows + header
    sty = sty.set_table_styles([
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
    ])

    sty = sty.applymap(highlight_placeholders)

    # Emphasize ticker column if present
    if "Ticker" in df.columns:
        sty = sty.set_properties(subset=["Ticker"], **{"color":"#60a5fa","font-weight":"700"})

    return sty

# ---------- UI helpers ----------
def pill_row(options: list[str], state_key: str, default: str):
    # Read current selection
    active = st.session_state.get(state_key, default)

    cols = st.columns(len(options))
    for i, lab in enumerate(options):
        # Render using CURRENT active; clicking updates state immediately and re-runs
        if cols[i].button(
            lab,
            key=f"{state_key}_{lab}",
            use_container_width=True,
            type=("primary" if lab == active else "secondary"),
        ):
            st.session_state[state_key] = lab
            st.rerun()

    # Return latest value (after possible re-run this will be the new one)
    return st.session_state.get(state_key, default)

def tab_row(options: list[str], state_key: str, default: str):
    active = st.session_state.get(state_key, default)

    cols = st.columns(len(options))
    for i, lab in enumerate(options):
        if cols[i].button(
            lab,
            key=f"{state_key}_{lab}",
            use_container_width=True,
            type=("primary" if lab == active else "secondary"),
        ):
            st.session_state[state_key] = lab
            st.rerun()

    return st.session_state.get(state_key, default)

def multiselect_with_chips_below(label, options, key):
    """
    Multiselect with centered 'All' / 'None' below.
    Defaults to no selections.
    """
    pending_key = f"{key}__pending"
    action = st.session_state.pop(pending_key, None)
    if action == "all":
        st.session_state[key] = list(options)
    elif action == "none":
        st.session_state[key] = []

    if key not in st.session_state:
        st.session_state[key] = []   # default: empty

    out = st.multiselect(label, options=options, key=key)

    left, center, right = st.columns([1, 0.48, 1])
    with center:
        c1, gap, c2 = st.columns([1, 0.18, 1])
        with c1:
            if st.button("All", key=f"{key}_all", use_container_width=True):
                st.session_state[pending_key] = "all"
                st.rerun()
        with c2:
            if st.button("None", key=f"{key}_none", use_container_width=True):
                st.session_state[pending_key] = "none"
                st.rerun()
    return out


def mc_slider_billions(values: pd.Series, key="flt_mc_b_range"):
    """
    Log-warped dual slider in billions. Lower bound is fixed at 0 so the left end
    always shows 0, regardless of the dataset's minimum. End labels are rendered
    under the left/right ends of the track.
    """
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty or float(s.max()) <= 0:
        st.caption("Market Cap (GBP) — no data available")
        st.session_state["flt_mc_abs"] = (None, None)
        return (None, None)

    vmin_display = 0.0           # force slider to start at zero
    vmax = float(s.max())

    # log warp (use x+1 so zero is safe); lower log bound is exactly 0
    lmin = 0.0                   # log10(0+1)
    lmax = np.log10(vmax + 1.0)

    def from_t(t: float) -> float:
        # map t in [0,1] → value in [0, vmax]
        return max(0.0, 10 ** (lmin + t * (lmax - lmin)) - 1.0)

    # slider runs from 0..1; small step for fine control at low end
    t_lo, t_hi = st.slider(
        "Market Cap (GBP)",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.001,
        key=key, format=""  # hide built-in labels; we draw our own
    )

    lo_abs, hi_abs = from_t(t_lo), from_t(t_hi)
    st.session_state["flt_mc_abs"] = (lo_abs, hi_abs)

    # end labels aligned to the ends of the track
    st.markdown(
        f"""
        <div style="
            display:flex; justify-content:space-between;
            margin-top:.35rem; font-weight:700; color:#ffffff;">
            <span>{humanize_mc(lo_abs)}</span>
            <span>{humanize_mc(hi_abs)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return lo_abs, hi_abs

def finviz_pager(total_rows: int, per_page: int, key: str = "page"):
    """Return (page, pages). Renders pager *below* the table FINVIZ-style.
       Forces a rerun immediately on click so paging is single-click.
    """
    pages = max(1, math.ceil(total_rows / per_page))
    page = int(st.session_state.get(key, 1))

    # Build visible labels: 1..5 and last (with ellipsis when needed)
    labels = list(range(1, min(6, pages + 1)))
    if pages > 6:
        labels += ["…", pages]

    clicked = False
    new_page = page

    st.markdown('<div class="pager">', unsafe_allow_html=True)
    cprev, *cnums, cnext = st.columns(len(labels) + 2)

    with cprev:
        if st.button("◀", key=f"{key}_prev", use_container_width=True, disabled=(page <= 1)):
            new_page = max(1, page - 1)
            clicked = True

    # Number buttons
    for i, lab in enumerate(labels):
        with cnums[i]:
            if lab == "…":
                st.write("…")
            else:
                btn = st.button(
                    str(lab),
                    key=f"{key}_{lab}",
                    type=("primary" if lab == page else "secondary"),
                    use_container_width=True,
                )
                if btn:
                    new_page = int(lab)
                    clicked = True

    with cnext:
        if st.button("▶", key=f"{key}_next", use_container_width=True, disabled=(page >= pages)):
            new_page = min(pages, page + 1)
            clicked = True

    st.markdown("</div>", unsafe_allow_html=True)

    # If a pager button was clicked, update and rerun immediately (so table changes now)
    if clicked:
        st.session_state[key] = new_page
        st.rerun()

    # Keep session state in sync
    st.session_state[key] = new_page
    return new_page, pages

# ---------------------------- Load & Build ----------------------------
data = load_all()
company = data.get("companyData", pd.DataFrame())
mktcap_gbp_wide = data.get("marketCapGBP", pd.DataFrame())

# --- Earnings Momentum: load precomputed table ---
em_ready_full = load_em_ready("data_private/earnings_momentum_ready.parquet")

header("", "Stock Screener", "")

universe_df = build_universe(company, mktcap_gbp_wide)

# --- Preloaded performance (decimal form) ---
from modules.performance import load_performance_ready, render_performance_section
perf_ready = load_performance_ready("data_private/performance_ready.parquet")



# ---------------------------- Filter options ----------------------------
st.markdown("#### Filter options")
FILTER_GROUPS = ["Descriptive","Fundamental","Technical","News","ETF","All"]
active_group  = pill_row(FILTER_GROUPS, "ui_active_group", "Descriptive")

if active_group in ("Descriptive","All"):
    c_sl, _sp = st.columns([0.25, 0.75])
    with c_sl:
        mc_lo, mc_hi = mc_slider_billions(universe_df["market_cap_gbp"])
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)  # small space under slider

    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
    with c1:
        sector_opts = sorted(universe_df["sector"].dropna().unique().tolist())
    with c2:
        ind_opts = sorted(universe_df["industry"].dropna().unique().tolist())
    with c3:
        ctry_opts = sorted(universe_df["country"].dropna().unique().tolist())

    # New options for ISIN and Exchange Name (from companyData / universe_df)
    isin_opts = sorted(universe_df["isin"].dropna().unique().tolist()) if "isin" in universe_df.columns else []
    exch_opts = sorted(universe_df["exchangeFullName"].dropna().unique().tolist()) if "exchangeFullName" in universe_df.columns else []

    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
    with c1:
        pick_sector   = multiselect_with_chips_below("Sector", sector_opts, key="flt_sector")
    with c2:
        pick_industry = multiselect_with_chips_below("Industry", ind_opts, key="flt_industry")
    with c3:
        pick_country  = multiselect_with_chips_below("Country", ctry_opts, key="flt_country")
    
    # New row directly BELOW Sector & Industry: ISIN and Exchange Name
    # --- Extra row: Exchange Name only ---
    exch_opts = sorted(
        universe_df.get("exchangeFullName", pd.Series(dtype=str))
                .dropna().astype(str).unique().tolist()
    )
    pick_exch = multiselect_with_chips_below("Exchange Name", exch_opts, key="flt_exchange")

    q = st.text_input("Search (Ticker or Company)", value=st.session_state.get("flt_q", ""), key="flt_q", placeholder="Type to filter…")

    if st.button("Reset Filters", type="secondary", key="btn_reset"):
        for k in list(st.session_state.keys()):
            if str(k).startswith("flt_") or k in ("ui_active_group", "ui_active_section", "page", "page_fingerprint"):
                st.session_state.pop(k, None)
        try: st.rerun()
        except Exception: st.experimental_rerun()

# ---------------------------- Data header + section tabs ----------------------------
st.markdown("### Data")
SECTIONS = ["Overview","Earnings Momentum","Performance"]
active_section = tab_row(SECTIONS, "ui_active_section", "Overview")

st.markdown("<hr style='border-color:#0b1220; opacity:.6; margin:.3rem 0 1.0rem 0'/>", unsafe_allow_html=True)

def get_filtered_universe_view() -> pd.DataFrame:
    """Use the current UI filters to produce the filtered universe view used by all sections."""
    return apply_overview_filters(universe_df)

# ---------------------------- Build Overview table ----------------------------
def apply_overview_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Market cap
    mc_abs = st.session_state.get("flt_mc_abs", (None, None))
    if mc_abs and mc_abs[0] is not None and mc_abs[1] is not None and "market_cap_gbp" in out.columns:
        lo, hi = mc_abs
        out = out[(out["market_cap_gbp"] >= lo) & (out["market_cap_gbp"] <= hi)]
    # Categorical
    if st.session_state.get("flt_sector"):
        out = out[out["sector"].isin(st.session_state["flt_sector"])]
    if st.session_state.get("flt_industry"):
        out = out[out["industry"].isin(st.session_state["flt_industry"])]
    if st.session_state.get("flt_country"):
        out = out[out["country"].isin(st.session_state["flt_country"])]

    if st.session_state.get("flt_exch"):
        out = out[out["exchangeFullName"].isin(st.session_state["flt_exch"])]

    # Search
    if st.session_state.get("flt_q"):
        q = str(st.session_state["flt_q"]).strip().lower()
        if q:
            out = out[(out["ticker"].str.lower().str.contains(q)) | (out["name"].str.lower().str.contains(q))]
    # Presentation
    out["Market Cap (GBP)"] = out["market_cap_gbp"].map(humanize_mc)

    cols = ["ticker","name","sector","industry","country","Market Cap (GBP)"]
    rename_map = {
        "ticker":"Ticker",
        "name":"Company",
        "sector":"Sector",
        "industry":"Industry",
        "country":"Country",
    }

    # Optional columns we want available in the “Columns to show” UI
    if "isin" in out.columns:
        cols.append("isin")
        rename_map["isin"] = "ISIN"

    if "exchangeFullName" in out.columns:
        cols.append("exchangeFullName")
        rename_map["exchangeFullName"] = "Exchange Name"

    view = out[cols].rename(columns=rename_map)
    return view.reset_index(drop=True)

if active_section == "Overview":
    if universe_df.empty:
        st.info("No company data available.")
    else:
        view_raw = apply_overview_filters(universe_df)

        # Column visibility (now includes ISIN / Exchange Name when present)
        default_cols = ["Ticker","Company","Sector","Industry","Country","Market Cap (GBP)"]

        # Build full options list from what view_raw actually has
        all_cols = [c for c in default_cols if c in view_raw.columns]
        for extra in ["ISIN", "Exchange Name"]:
            if extra in view_raw.columns and extra not in all_cols:
                all_cols.append(extra)

        # Keep prior selection if user changed it before
        prev_sel = st.session_state.get("vis_cols")
        init_sel = [c for c in (prev_sel or default_cols) if c in all_cols]

        cols_pick = st.multiselect("Columns to show", options=all_cols, default=init_sel, key="vis_cols")
        view = view_raw[cols_pick] if cols_pick else view_raw[all_cols]

        # Sorting controls (removed toggle; simple order chooser)
        cA, cB = st.columns([0.7, 0.3])
        with cA:
            sort_col_opts = view.columns.tolist()
            sort_col = st.selectbox(
                "Sort by",
                options=sort_col_opts,
                index=(sort_col_opts.index("Ticker") if "Ticker" in sort_col_opts else 0),
                key="sort_col"
            )
        with cB:
            order = st.selectbox("Order", ["Ascending","Descending"],
                                 index=(1 if sort_col == "Market Cap (GBP)" else 0),
                                 key="sort_order")
            sort_asc = (order == "Ascending")

        per_page = st.selectbox("Rows per page", [20, 50, 100, 200], index=0, key="per_page")

        # Reset pagination if filter/sort changed
        key_parts = (
            str(st.session_state.get("flt_mc_abs")),
            str(st.session_state.get("flt_sector")),
            str(st.session_state.get("flt_industry")),
            str(st.session_state.get("flt_country")),
            str(st.session_state.get("flt_q")),
            sort_col, str(sort_asc), str(per_page), ",".join(view.columns)
        )
        change_fingerprint = "||".join(key_parts)
        if st.session_state.get("page_fingerprint") != change_fingerprint:
            st.session_state["page"] = 1
            st.session_state["page_fingerprint"] = change_fingerprint

        # Sort ENTIRE filtered dataset before pagination
        sort_key = "market_cap_gbp" if sort_col == "Market Cap (GBP)" else sort_col
        try:
            to_sort = view_raw.join(universe_df.set_index("ticker")["market_cap_gbp"], on="Ticker", how="left")
            sorted_full = to_sort.sort_values(sort_key, ascending=sort_asc, kind="mergesort")
            view_sorted = sorted_full[view.columns]
        except Exception:
            view_sorted = view

        # Page AFTER sorting
        total = len(view_sorted)
        page = int(st.session_state.get("page", 1))
        pages = max(1, math.ceil(total / per_page))
        start, end = (page - 1) * per_page, (page - 1) * per_page + per_page
        page_df = view_sorted.iloc[start:end].copy()
        page_df.insert(0, "No.", range(start + 1, start + 1 + len(page_df)))

        # Table
        if page_df.empty:
            st.info("No rows match your filters.")
        else:
            try:
                st.table(style_table(page_df).hide(axis="index"))
            except Exception:
                st.table(style_table(page_df))
            export_buttons(view_sorted, filename="overview")

        # FINVIZ-style pager UNDER the table
        st.write("")  # small spacer
        page, pages = finviz_pager(total, per_page, key="page")
        st.caption(f"Overview | rows={total} | pages={pages} | showing {len(page_df)}")

if active_section == "Performance":
    base_view = get_filtered_universe_view()
    base = base_view[["Ticker", "Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")

    if perf_ready is None or perf_ready.empty:
        st.info("Performance data file is missing or empty.")
    else:
        # Filter to the current universe selection
        perf_core = perf_ready[perf_ready["Ticker"].isin(base["Ticker"])].copy()

        # Prefer Company names from the current universe selection when available
        comp_map = base.set_index("Ticker")["Company"].to_dict()
        if "Company" in perf_core.columns:
            perf_core["Company"] = perf_core["Ticker"].map(comp_map).fillna(perf_core["Company"])

        # Reorder to match the base selection order
        order_map = {t: i for i, t in enumerate(base["Ticker"])}
        perf_core["__order__"] = perf_core["Ticker"].map(order_map)
        perf_core = perf_core.sort_values("__order__").drop(columns="__order__")

        render_performance_section(
            perf_core,
            style_table_fn=style_table,
            pager_fn=finviz_pager,
            export_buttons_fn=export_buttons,
        )

# ---------------------------- Earnings Momentum ----------------------------
if active_section == "Earnings Momentum":
    # Use the SAME filters as Overview
    filtered_universe_view = apply_overview_filters(universe_df)
    base_em = filtered_universe_view[["Ticker","Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")

    # Filter the precomputed em_ready to the currently filtered universe
    em_core = build_earnings_momentum_table(base_em, em_ready_full)

    render_earnings_momentum_section(
        em_core,
        style_table_fn=style_table,
        pager_fn=finviz_pager,
        export_buttons_fn=export_buttons,
        key_prefix="em_",
    )