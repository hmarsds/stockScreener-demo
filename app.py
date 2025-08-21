# app.py — L/S Stock Screener (Overview + Earnings Momentum)
import math
import numpy as np
import pandas as pd
import streamlit as st
from utils.auth import require_access_code
from pathlib import Path
import re
from datetime import datetime
import os

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
from modules.valuation import (
    load_valuation_ready,
    build_valuation_table,
    render_valuation_section,
)
from modules.quality import (
    load_quality_ready,
    build_quality_table,
    render_quality_section,
)
from modules.technicals import load_technicals_ready, render_technicals_section

# ---------- Root path fix (run-anywhere) ----------
# Ensure all relative paths like "data_private/..." resolve relative to this file.
_APP_DIR = Path(__file__).resolve().parent
try:
    os.chdir(_APP_DIR)
except Exception:
    pass


# ---------------------------- Page / Theme ----------------------------
st.set_page_config(page_title="L/S Stock Screener", page_icon="🕵️‍♂️", layout="wide")
inject_dark_theme()

# It restores any slider wrappers your dark.css might hide, and hides ONLY the tiny labels above.



# ---------- Auth Gate ----------
_auth_box = st.empty()
with _auth_box.container():
    if not require_access_code():
        st.stop()
_auth_box.empty()


# --- Slider visibility + hide min/max labels ---
# --- Sliders: restore wrappers; hide tiny 0/1 labels above ---
st.markdown("""
<style id="slider-restore">
/* Make sure inner wrappers are visible on all Streamlit versions */
.stSlider [data-baseweb="slider"] > div { display:block !important; }
.stSlider [data-baseweb="slider"] + div { display:block !important; }
[data-testid="stSlider"]{ min-height:46px !important; }

/* Compact look + visible handle */
.stSlider [data-baseweb="slider"]{ height:12px !important; padding:8px 0 !important; }
.stSlider [role="slider"]{
  width:18px !important; height:18px !important;
  background:#ffffff !important; border:2px solid #22c55e !important; box-shadow:none !important;
}

/* Hide the tiny min/max/ticks above the track */
.stSlider [data-testid="stTickBar"],
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"]{ display:none !important; }
</style>
""", unsafe_allow_html=True)

# ============================ Cache helpers ============================
def _mtime(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0

@st.cache_data(show_spinner=False, ttl=3600)
def get_all_data_cached():
    return load_all()

@st.cache_data(show_spinner=False, ttl=3600)
def get_ready_cached(path: str, kind: str):
    """Cache wrapper around the ready loaders (invalidates when file mtime changes)."""
    _ = _mtime(path)
    if kind == "em":   return load_em_ready(path)
    if kind == "val":  return load_valuation_ready(path)
    if kind == "qual": return load_quality_ready(path)
    if kind == "perf": return load_performance_ready(path)
    raise ValueError(kind)

@st.cache_data(show_spinner=False, ttl=3600)
def get_price_series_for_date(path: str, date_str: str):
    """Return (Series[ticker->closeGBP], vmax) for a single date."""
    _ = _mtime(path)
    mp = load_close_gbp_map_for_date(path, date_str)
    s = pd.Series(mp, dtype="float64")
    vmax = float(s.max()) if not s.empty else 0.0
    return s, vmax

@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamentals_df(val_df: pd.DataFrame | None, qlt_df: pd.DataFrame | None):
    """Merge numeric Valuation + Quality columns for Fundamental filters UI."""
    def _num(df):
        return [c for c in df.columns if c not in ("Ticker", "Company") and pd.api.types.is_numeric_dtype(df[c])]
    out = pd.DataFrame({"Ticker": []})
    if val_df is not None and not val_df.empty:
        out = val_df[["Ticker"] + _num(val_df)].copy()
    if qlt_df is not None and not qlt_df.empty:
        to_add = qlt_df[["Ticker"] + _num(qlt_df)].copy()
        out = to_add if out.empty else out.merge(to_add, how="outer", on="Ticker")
    return out

# ============================ Helpers ============================
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

# ---- Close Price (GBP) loader (one date only) ----
def load_close_gbp_map_for_date(path: str | Path, date_str: str = "2025-08-18") -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        df.index = pd.to_datetime(df.index, errors="coerce")
    df.columns = [str(c).upper().strip() for c in df.columns]

    target = pd.Timestamp(date_str)
    if target not in df.index:
        return {}
    row = pd.to_numeric(df.loc[target], errors="coerce")
    ser = row.dropna()
    ser.index = ser.index.astype(str)
    return ser.to_dict()

# ---- Sliders (log-warped) ----
def price_slider_gbp_log(vmax: float, key="flt_price_abs"):
    if not vmax or vmax <= 0:
        st.caption("Close Price (GBP) — no data available")
        st.session_state[key] = (None, None)
        return (None, None)

    lmin, lmax = 0.0, np.log10(vmax + 1.0)
    def from_t(t: float) -> float:
        return max(0.0, 10 ** (lmin + t * (lmax - lmin)) - 1.0)

    st.markdown("**Close Price (GBP)**")
    t_lo, t_hi = st.slider(
        "", min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.001,
        key=key + "__t",
        label_visibility="collapsed",
        format=" "  # <--- makes built-in value text blank
    )
    vlo, vhi = from_t(t_lo), from_t(t_hi)
    st.session_state[key] = (vlo, vhi)
    st.markdown(
        '<div style="display:flex;justify-content:space-between;'
        'margin-top:.35rem;font-weight:700;color:#ffffff;">'
        f'<span>{vlo:.2f}</span><span>{vhi:.2f}</span></div>',
        unsafe_allow_html=True,
    )
    return vlo, vhi

def mc_slider_billions(values: pd.Series, key="flt_mc_b_range"):
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty or float(s.max()) <= 0:
        st.caption("Market Cap (GBP) — no data available")
        st.session_state["flt_mc_abs"] = (None, None)
        return (None, None)

    vmax = float(s.max())
    lmin, lmax = 0.0, np.log10(vmax + 1.0)
    def from_t(t: float) -> float:
        return max(0.0, 10 ** (lmin + t * (lmax - lmin)) - 1.0)

    st.markdown("**Market Cap (GBP)**")
    t_lo, t_hi = st.slider(
        "", min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.001,
        key=key + "__t",
        label_visibility="collapsed",
        format=" "  # <--- makes built-in value text blank
    )
    lo_abs, hi_abs = from_t(t_lo), from_t(t_hi)
    st.session_state["flt_mc_abs"] = (lo_abs, hi_abs)
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;'
        f'margin-top:.35rem;font-weight:700;color:#ffffff;">'
        f'<span>{humanize_mc(lo_abs)}</span><span>{humanize_mc(hi_abs)}</span></div>',
        unsafe_allow_html=True,
    )
    return lo_abs, hi_abs

# ---------- UI helpers ----------
def pill_row(options: list[str], state_key: str, default: str):
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
    pending_key = f"{key}__pending"
    action = st.session_state.pop(pending_key, None)
    if action == "all":
        st.session_state[key] = list(options)
    elif action == "none":
        st.session_state[key] = []
    if key not in st.session_state:
        st.session_state[key] = []

    out = st.multiselect(label, options=options, key=key)
    left, center, right = st.columns([1, 0.48, 1])
    with center:
        c1, gap, c2 = st.columns([1, 0.18, 1])
        with c1:
            if st.button("All", key=f"{key}_all", use_container_width=True):
                st.session_state[pending_key] = "all"; st.rerun()
        with c2:
            if st.button("None", key=f"{key}_none", use_container_width=True):
                st.session_state[pending_key] = "none"; st.rerun()
    return out

def finviz_pager(total_rows: int, per_page: int, key: str = "page"):
    pages = max(1, math.ceil(total_rows / per_page))
    page = int(st.session_state.get(key, 1))
    page = max(1, min(page, pages))

    def visible_pages(cur: int, total: int, radius: int = 2):
        if total <= 10:
            return list(range(1, total + 1))
        left  = max(2, cur - radius)
        right = min(total - 1, cur + radius)
        items = [1]
        if left > 2:
            items.append("…")
        items.extend(range(left, right + 1))
        if right < total - 1:
            items.append("…")
        items.append(total)
        return items

    labels = visible_pages(page, pages)
    clicked, new_page = False, page

    st.markdown('<div class="pager">', unsafe_allow_html=True)
    cprev, *cnums, cnext = st.columns(len(labels) + 2)

    with cprev:
        if st.button("◀", key=f"{key}_prev", use_container_width=True, disabled=(page <= 1)):
            new_page = max(1, page - 1); clicked = True

    for i, lab in enumerate(labels):
        with cnums[i]:
            if lab == "…":
                st.write("…")
            else:
                btn = st.button(
                    str(lab),
                    key=f"{key}_{lab}",
                    type=("primary" if int(lab) == page else "secondary"),
                    use_container_width=True,
                )
                if btn:
                    new_page = int(lab); clicked = True

    with cnext:
        if st.button("▶", key=f"{key}_next", use_container_width=True, disabled=(page >= pages)):
            new_page = min(pages, page + 1); clicked = True

    st.markdown("</div>", unsafe_allow_html=True)
    if clicked:
        st.session_state[key] = new_page
        st.rerun()
    st.session_state[key] = new_page
    return new_page, pages

def style_table(df: pd.DataFrame):
    sty = df.style.set_properties(**{"color":"#e5e7eb","border-color":"#1f2937"})
    def _cell(val): return "text-align: center;" if val == "•" else "text-align: right;"
    sty = sty.set_table_styles([
        {"selector": "thead th", "props": [("background-color","#0f172a"),("color","#f9fafb"),("border-color","#1f2937")]},
        {"selector": "tbody td", "props": [("border-color","#1f2937")]},
        {"selector": "tbody tr:nth-child(odd) td", "props": [("background-color","#0f172a")]},
        {"selector": "tbody tr:nth-child(even) td", "props": [("background-color","#111827")]},
    ]).applymap(_cell)
    if "Ticker" in df.columns:
        sty = sty.set_properties(subset=["Ticker"], **{"color":"#60a5fa","font-weight":"700"})
    return sty

# ============================ Build universe & load data ============================
data = get_all_data_cached()
company_df   = data.get("companyData", pd.DataFrame())
mktcap_gbp_w = data.get("marketCapGBP", pd.DataFrame())

# --- optional diagnostics (safe) ---
from glob import glob
if st.sidebar.checkbox("🔎 Data diagnostics", False, key="diag_toggle"):
    st.write("CWD:", Path.cwd())
    st.write("data_private exists:", Path("data_private").exists())
    st.write("Parquet files in data_private:",
             sorted([Path(p).name for p in glob("data_private/*.parquet")]))
    st.write("companyData:", "EMPTY" if company_df is None or company_df.empty else company_df.shape)
    st.write("marketCapGBP:", "EMPTY" if mktcap_gbp_w is None or mktcap_gbp_w.empty else mktcap_gbp_w.shape)

header("", "Stock Screener", "")

def build_universe(company_df: pd.DataFrame, mcap_gbp_wide: pd.DataFrame | None) -> pd.DataFrame:
    if company_df is None or company_df.empty:
        return pd.DataFrame(columns=[
            "ticker","name","sector","industry","country","currency",
            "isin","exchangeFullName","market_cap_gbp"
        ])
    df = company_df.rename(columns={"symbol":"ticker","companyName":"name"}).copy()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    for c in ("ticker","name","sector","industry","country","currency","isin","exchangeFullName"):
        if c not in df.columns: df[c] = pd.NA

    if isinstance(df["ticker"], pd.DataFrame):
        df["ticker"] = df["ticker"].iloc[:,0]
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"]   = df["name"].astype(str).str.strip()
    for c in ("sector","industry","currency","isin","exchangeFullName","country"):
        df[c] = df[c].astype(str).str.strip()

    df = df[["ticker","name","sector","industry","country","currency","isin","exchangeFullName"]] \
           .dropna(subset=["ticker"]).drop_duplicates("ticker")

    if mcap_gbp_wide is not None and not mcap_gbp_wide.empty:
        df["market_cap_gbp"] = df["ticker"].map(latest_non_null_map_from_wide(mcap_gbp_wide))
    else:
        df["market_cap_gbp"] = np.nan

    df["country"] = df["country"].map(country_to_name)
    return df

universe_df = build_universe(company_df, mktcap_gbp_w)

# Ready datasets (cached)
em_ready_full        = get_ready_cached("data_private/earnings_momentum_ready.parquet", "em")
valuation_ready_full = get_ready_cached("data_private/valuation_ready.parquet", "val")
quality_ready_full   = get_ready_cached("data_private/quality_ready.parquet", "qual")
perf_ready           = get_ready_cached("data_private/performance_ready.parquet", "perf")
tech_ready = load_technicals_ready("data_private")

# Close prices for 18-Aug-2025 (cached, already single-day parquet)
price_series, price_vmax = get_price_series_for_date("data_private/pricesGBP_18.parquet", "2025-08-18")
if not price_series.empty:
    universe_df = universe_df.merge(
        price_series.rename("close_price_gbp"),
        left_on="ticker", right_index=True, how="left"
    )
close_price_map = price_series.to_dict()  # for optional column add

# Fundamentals data-frame + metric names
_fundamentals_df = get_fundamentals_df(valuation_ready_full, quality_ready_full)
VAL_METRICS = [c for c in (valuation_ready_full.columns if valuation_ready_full is not None else []) if c not in ("Ticker","Company")]
QLT_METRICS = [c for c in (quality_ready_full.columns   if quality_ready_full   is not None else []) if c not in ("Ticker","Company")]

# ============================ Filter options ============================
st.markdown("#### Filter options")
FILTER_GROUPS = ["Descriptive","Fundamental","All"]
active_group  = pill_row(FILTER_GROUPS, "ui_active_group", "Descriptive")

if active_group in ("Descriptive","All"):
    # Two aligned sliders with a clear spacer between
    c_mc, c_gap, c_px = st.columns([0.44, 0.12, 0.44])
    with c_mc:
        mc_lo, mc_hi = mc_slider_billions(universe_df["market_cap_gbp"])
    with c_px:
        px_lo, px_hi = price_slider_gbp_log(price_vmax, key="flt_price_abs")

    # Categorical filters
    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
    with c1:
        sector_opts = sorted(universe_df["sector"].dropna().unique().tolist())
    with c2:
        ind_opts = sorted(universe_df["industry"].dropna().unique().tolist())
    with c3:
        ctry_opts = sorted(universe_df["country"].dropna().unique().tolist())

    c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
    with c1:
        pick_sector   = multiselect_with_chips_below("Sector", sector_opts, key="flt_sector")
    with c2:
        pick_industry = multiselect_with_chips_below("Industry", ind_opts, key="flt_industry")
    with c3:
        pick_country  = multiselect_with_chips_below("Country", ctry_opts, key="flt_country")

    exch_opts = sorted(universe_df.get("exchangeFullName", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    pick_exch = multiselect_with_chips_below("Exchange Name", exch_opts, key="flt_exchange")

    q = st.text_input("Search (Ticker or Company)", value=st.session_state.get("flt_q", ""), key="flt_q", placeholder="Type to filter…")

    if st.button("Reset Filters", type="secondary", key="btn_reset"):
        for k in list(st.session_state.keys()):
            if str(k).startswith("flt_") or k in ("ui_active_group", "ui_active_section", "page", "page_fingerprint"):
                st.session_state.pop(k, None)
        st.rerun()

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def _parse_num(s: str):
    try: return float(str(s).strip())
    except Exception: return None

def _get_metric_hint(df: pd.DataFrame, col: str) -> tuple[float|None, float|None]:
    if df is None or df.empty or col not in df.columns: return (None, None)
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty: return (None, None)
    return (float(s.quantile(0.05)), float(s.quantile(0.95)))

def _render_minmax_row(label: str, df_ref: pd.DataFrame, key_prefix: str):
    k_min = f"ff_{key_prefix}_min"; k_max = f"ff_{key_prefix}_max"
    hint_lo, hint_hi = _get_metric_hint(df_ref, label)
    c1, c2, c3 = st.columns([0.30, 0.35, 0.35])
    with c1: st.caption(label)
    with c2: st.text_input("Min", key=k_min, placeholder=("e.g. " + (f"{hint_lo:.2f}" if hint_lo is not None else "")))
    with c3: st.text_input("Max", key=k_max, placeholder=("e.g. " + (f"{hint_hi:.2f}" if hint_hi is not None else "")))
    return (_parse_num(st.session_state.get(k_min, "")), _parse_num(st.session_state.get(k_max, "")))

def _current_fund_filters() -> dict:
    out = {}
    for col in (VAL_METRICS + QLT_METRICS):
        key = _slug(col)
        vmin = _parse_num(st.session_state.get(f"ff_{key}_min", ""))
        vmax = _parse_num(st.session_state.get(f"ff_{key}_max", ""))
        if vmin is not None or vmax is not None:
            out[col] = (vmin, vmax)
    return out

def _clear_fund_filters():
    for col in (VAL_METRICS + QLT_METRICS):
        key = _slug(col)
        st.session_state[f"ff_{key}_min"] = ""
        st.session_state[f"ff_{key}_max"] = ""

def _apply_fundamental_mask(tickers: pd.Series) -> pd.Series:
    ff = _current_fund_filters()
    if not ff or _fundamentals_df is None or _fundamentals_df.empty:
        return pd.Series(True, index=tickers.index)
    df = pd.DataFrame({"Ticker": tickers.astype(str)}).merge(_fundamentals_df, how="left", on="Ticker")
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in ff.items():
        vals = pd.to_numeric(df.get(col), errors="coerce")
        if lo is not None: mask &= vals.ge(lo)
        if hi is not None: mask &= vals.le(hi)
        if lo is not None or hi is not None: mask &= vals.notna()
    return mask.reindex(index=tickers.index).fillna(False)

# ---------------- Fundamentals UI ----------------
if active_group in ("Fundamental","All"):
    st.markdown("##### Fundamental filters")
    cc1, cc2 = st.columns([0.85, 0.15])
    with cc2:
        if st.button("Clear all", use_container_width=True):
            _clear_fund_filters(); st.rerun()

    if VAL_METRICS:
        st.markdown("**Valuation**")
        for col in VAL_METRICS:
            _render_minmax_row(col, _fundamentals_df, _slug(col))

    if QLT_METRICS:
        st.markdown("**Quality**")
        for col in QLT_METRICS:
            _render_minmax_row(col, _fundamentals_df, _slug(col))

# ============================ Data header + section tabs ============================
st.markdown("### Data")
SECTIONS = ["Overview","Earnings Momentum","Performance","Valuation","Quality","Technicals"]
active_section = tab_row(SECTIONS, "ui_active_section", "Overview")
st.markdown("<hr style='border-color:#0b1220; opacity:.6; margin:.3rem 0 1.0rem 0'/>", unsafe_allow_html=True)

def apply_overview_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Market cap filter
    mc_abs = st.session_state.get("flt_mc_abs", (None, None))
    if mc_abs and mc_abs[0] is not None and mc_abs[1] is not None and "market_cap_gbp" in out.columns:
        lo, hi = mc_abs
        out = out[(out["market_cap_gbp"] >= lo) & (out["market_cap_gbp"] <= hi)]

    # Close Price (GBP) filter (from slider)
    pr_abs = st.session_state.get("flt_price_abs", (None, None))
    if pr_abs and pr_abs[0] is not None and pr_abs[1] is not None and "close_price_gbp" in out.columns:
        lo, hi = pr_abs
        out = out[(out["close_price_gbp"] >= lo) & (out["close_price_gbp"] <= hi)]

    # Categorical filters
    if st.session_state.get("flt_sector"):
        out = out[out["sector"].isin(st.session_state["flt_sector"])]
    if st.session_state.get("flt_industry"):
        out = out[out["industry"].isin(st.session_state["flt_industry"])]
    if st.session_state.get("flt_country"):
        out = out[out["country"].isin(st.session_state["flt_country"])]
    if st.session_state.get("flt_exchange"):
        out = out[out["exchangeFullName"].isin(st.session_state["flt_exchange"])]

    # Search
    if st.session_state.get("flt_q"):
        q = str(st.session_state["flt_q"]).strip().lower()
        if q:
            out = out[(out["ticker"].str.lower().str.contains(q)) | (out["name"].str.lower().str.contains(q))]

    # Presentation fields
    out["Market Cap (GBP)"] = out["market_cap_gbp"].map(humanize_mc)
    if "close_price_gbp" in out.columns:
        out["Close Price (GBP)"] = pd.to_numeric(out["close_price_gbp"], errors="coerce").round(2)

    cols = ["ticker","name","sector","industry","country","Market Cap (GBP)"]
    rename_map = {"ticker":"Ticker","name":"Company","sector":"Sector","industry":"Industry","country":"Country"}
    if "Close Price (GBP)" in out.columns:
        cols.append("Close Price (GBP)")
    if "isin" in out.columns:
        cols.append("isin"); rename_map["isin"] = "ISIN"
    if "exchangeFullName" in out.columns:
        cols.append("exchangeFullName"); rename_map["exchangeFullName"] = "Exchange Name"

    view = out[cols].rename(columns=rename_map)
    return view.reset_index(drop=True)

def get_filtered_universe_view() -> pd.DataFrame:
    return apply_overview_filters(universe_df)

# ---------------------------- Overview ----------------------------
if active_section == "Overview":
    if universe_df.empty:
        st.info("No company data available.")
    else:
        view_raw = apply_overview_filters(universe_df)

        if active_group in ("Fundamental","All") and not view_raw.empty:
            mask = _apply_fundamental_mask(view_raw["Ticker"])
            view_raw = view_raw.loc[mask].reset_index(drop=True)

        default_cols = ["Ticker","Company","Sector","Industry","Country","Market Cap (GBP)"]
        extras_candidates = ["ISIN", "Exchange Name", "Close Price (GBP)"]
        all_cols = [c for c in default_cols if c in view_raw.columns]
        all_cols += [c for c in extras_candidates if c in view_raw.columns and c not in all_cols]

        prev_sel = st.session_state.get("vis_cols")
        init_sel = [c for c in (prev_sel or default_cols) if c in all_cols]
        cols_pick = st.multiselect("Columns to show", options=all_cols, default=init_sel, key="vis_cols")
        view = view_raw[cols_pick] if cols_pick else view_raw[all_cols]

        cA, cB = st.columns([0.7, 0.3])
        with cA:
            sort_col_opts = view.columns.tolist()
            sort_col = st.selectbox("Sort by", options=sort_col_opts,
                                    index=(sort_col_opts.index("Ticker") if "Ticker" in sort_col_opts else 0),
                                    key="sort_col")
        with cB:
            order = st.selectbox("Order", ["Ascending","Descending"],
                                 index=(1 if sort_col == "Market Cap (GBP)" else 0), key="sort_order")
            sort_asc = (order == "Ascending")

        per_page = st.selectbox("Rows per page", [20, 50, 100, 200], index=0, key="per_page")

        key_parts = (
            str(st.session_state.get("flt_mc_abs")),
            str(st.session_state.get("flt_sector")),
            str(st.session_state.get("flt_industry")),
            str(st.session_state.get("flt_country")),
            str(st.session_state.get("flt_exchange")),
            str(st.session_state.get("flt_q")),
            sort_col, str(sort_asc), str(per_page), ",".join(view.columns)
        )
        change_fingerprint = "||".join(key_parts)
        if st.session_state.get("page_fingerprint") != change_fingerprint:
            st.session_state["page"] = 1
            st.session_state["page_fingerprint"] = change_fingerprint

        sort_key = "market_cap_gbp" if sort_col == "Market Cap (GBP)" else sort_col
        try:
            to_sort = view_raw.join(universe_df.set_index("ticker")["market_cap_gbp"], on="Ticker", how="left")
            view_sorted = to_sort.sort_values(sort_key, ascending=sort_asc, kind="mergesort")[view.columns]
        except Exception:
            view_sorted = view

        total = len(view_sorted)
        page = int(st.session_state.get("page", 1))
        pages = max(1, math.ceil(total / per_page))
        start, end = (page - 1) * per_page, (page - 1) * per_page + per_page
        page_df = view_sorted.iloc[start:end].copy()
        page_df.insert(0, "No.", range(start + 1, start + 1 + len(page_df)))

        if page_df.empty:
            st.info("No rows match your filters.")
        else:
            try: st.table(style_table(page_df).hide(axis="index"))
            except Exception: st.table(style_table(page_df))
            export_buttons(view_sorted, filename="overview")

        st.write("")
        page, pages = finviz_pager(total, per_page, key="page")
        st.caption(f"Overview | rows={total} | pages={pages} | showing {len(page_df)}")

# ---------------------------- Performance ----------------------------
if active_section == "Performance":
    base_view = get_filtered_universe_view()
    base = base_view[["Ticker","Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")

    if perf_ready is None or perf_ready.empty:
        st.info("Performance data file is missing or empty.")
    else:
        if active_group in ("Fundamental","All") and not base.empty:
            m = _apply_fundamental_mask(base["Ticker"]); base = base.loc[m]

        perf_core = perf_ready[perf_ready["Ticker"].isin(base["Ticker"])].copy()

        comp_map = base.set_index("Ticker")["Company"].to_dict()
        if "Company" in perf_core.columns:
            perf_core["Company"] = perf_core["Ticker"].map(comp_map).fillna(perf_core["Company"])

        order_map = {t: i for i, t in enumerate(base["Ticker"])}
        perf_core["__order__"] = perf_core["Ticker"].map(order_map)
        perf_core = perf_core.sort_values("__order__").drop(columns="__order__")

        # Optional meta columns (Sector / Industry / Market Cap (GBP) / Close Price (GBP))
        def attach_meta_columns(df: pd.DataFrame, universe_df: pd.DataFrame, price_map: dict[str, float] | None = None) -> pd.DataFrame:
            if df is None or df.empty: return df
            meta = universe_df[["ticker","sector","industry","market_cap_gbp"]].copy()
            meta = meta.rename(columns={"ticker":"Ticker","sector":"Sector","industry":"Industry"})
            meta["Market Cap (GBP)"] = meta["market_cap_gbp"].map(humanize_mc)
            meta = meta.drop(columns=["market_cap_gbp"])
            if price_map:
                meta["Close Price (GBP)"] = meta["Ticker"].map(price_map)
            return df.merge(meta, on="Ticker", how="left")

        perf_core = attach_meta_columns(perf_core, universe_df, price_map=close_price_map)

        render_performance_section(
            perf_core,
            style_table_fn=style_table,
            pager_fn=finviz_pager,
            export_buttons_fn=export_buttons,
        )

# ---------------------------- Earnings Momentum ----------------------------
if active_section == "Earnings Momentum":
    filtered_universe_view = apply_overview_filters(universe_df)
    base_em = filtered_universe_view[["Ticker","Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")

    if active_group in ("Fundamental","All") and not base_em.empty:
        m = _apply_fundamental_mask(base_em["Ticker"]); base_em = base_em.loc[m]

    em_core = build_earnings_momentum_table(base_em, em_ready_full)

    # Optional meta columns
    def attach_meta_columns(df: pd.DataFrame, universe_df: pd.DataFrame, price_map: dict[str, float] | None = None) -> pd.DataFrame:
        if df is None or df.empty: return df
        meta = universe_df[["ticker","sector","industry","market_cap_gbp"]].copy()
        meta = meta.rename(columns={"ticker":"Ticker","sector":"Sector","industry":"Industry"})
        meta["Market Cap (GBP)"] = meta["market_cap_gbp"].map(humanize_mc)
        meta = meta.drop(columns=["market_cap_gbp"])
        if price_map:
            meta["Close Price (GBP)"] = meta["Ticker"].map(price_map)
        return df.merge(meta, on="Ticker", how="left")

    em_core = attach_meta_columns(em_core, universe_df, price_map=close_price_map)

    render_earnings_momentum_section(
        em_core,
        style_table_fn=style_table,
        pager_fn=finviz_pager,
        export_buttons_fn=export_buttons,
        key_prefix="em_",
    )

# ---------------------------- Valuation ----------------------------
if active_section == "Valuation":
    filtered_universe_view = apply_overview_filters(universe_df)
    base_val = filtered_universe_view[["Ticker","Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")
    if active_group in ("Fundamental","All") and not base_val.empty:
        m = _apply_fundamental_mask(base_val["Ticker"]); base_val = base_val.loc[m]

    val_core = build_valuation_table(base_val, valuation_ready_full)

    # Optional meta columns
    def attach_meta_columns(df: pd.DataFrame, universe_df: pd.DataFrame, price_map: dict[str, float] | None = None) -> pd.DataFrame:
        if df is None or df.empty: return df
        meta = universe_df[["ticker","sector","industry","market_cap_gbp"]].copy()
        meta = meta.rename(columns={"ticker":"Ticker","sector":"Sector","industry":"Industry"})
        meta["Market Cap (GBP)"] = meta["market_cap_gbp"].map(humanize_mc)
        meta = meta.drop(columns=["market_cap_gbp"])
        if price_map:
            meta["Close Price (GBP)"] = meta["Ticker"].map(price_map)
        return df.merge(meta, on="Ticker", how="left")

    val_core = attach_meta_columns(val_core, universe_df, price_map=close_price_map)

    render_valuation_section(
        val_core,
        style_table_fn=style_table,
        pager_fn=finviz_pager,
        export_buttons_fn=export_buttons,
        key_prefix="val_",
    )

# ---------------------------- Quality ----------------------------
if active_section == "Quality":
    filtered_universe_view = apply_overview_filters(universe_df)
    base_q = filtered_universe_view[["Ticker","Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")
    if active_group in ("Fundamental","All") and not base_q.empty:
        m = _apply_fundamental_mask(base_q["Ticker"]); base_q = base_q.loc[m]

    q_core = build_quality_table(base_q, quality_ready_full)

    def attach_meta_columns(df: pd.DataFrame, universe_df: pd.DataFrame, price_map: dict[str, float] | None = None) -> pd.DataFrame:
        if df is None or df.empty: return df
        meta = universe_df[["ticker","sector","industry","market_cap_gbp"]].copy()
        meta = meta.rename(columns={"ticker":"Ticker","sector":"Sector","industry":"Industry"})
        meta["Market Cap (GBP)"] = meta["market_cap_gbp"].map(humanize_mc)
        meta = meta.drop(columns=["market_cap_gbp"])
        if price_map:
            meta["Close Price (GBP)"] = meta["Ticker"].map(price_map)
        return df.merge(meta, on="Ticker", how="left")

    q_core = attach_meta_columns(q_core, universe_df, price_map=close_price_map)

    render_quality_section(
        q_core,
        style_table_fn=style_table,
        pager_fn=finviz_pager,
        export_buttons_fn=export_buttons,
        key_prefix="qlt_",
    )

# ---------------------------- Technicals ----------------------------
# ---------------------------- Technicals ----------------------------
if active_section == "Technicals":
    # start with the filtered universe (same as other sections)
    filtered_universe_view = apply_overview_filters(universe_df)
    base_t = filtered_universe_view[["Ticker","Company"]].dropna(subset=["Ticker"]).drop_duplicates("Ticker")

    # optional fundamental mask (to mirror other sections’ behavior)
    if active_group in ("Fundamental","All") and not base_t.empty:
        m = _apply_fundamental_mask(base_t["Ticker"])
        base_t = base_t.loc[m]

    # hand off to the Technicals section (no 'base_universe' kwarg)
    render_technicals_section(
        base_view=base_t,
        style_table_fn=style_table,
        pager_fn=finviz_pager,
        export_buttons_fn=export_buttons,
        key_prefix="tech_",
    )