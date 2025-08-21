# utils/screener_helpers.py
import numpy as np
import pandas as pd
import streamlit as st

ID_ORDER = ["ticker", "name", "isin", "sector", "industry", "country", "currency", "market_cap"]

def _ensure_cols(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def id_first_cols(df: pd.DataFrame):
    df = _ensure_cols(df, ID_ORDER)
    left = [c for c in ID_ORDER if c in df.columns]
    right = [c for c in df.columns if c not in left]
    return left + right

def section_pills(sections, active: str | None = None):
    """
    Horizontal section buttons (Overview, Earnings Momentum, …).
    Active section label is bolded for subtle emphasis (CSS handles color).
    """
    cols = st.columns(len(sections))
    clicked = None
    for i, sec in enumerate(sections):
        label = f"**{sec}**" if active and sec == active else sec
        if cols[i].button(label, use_container_width=True, key=f"pill_{sec}"):
            clicked = sec
    return clicked

def universe_toolbar(company_df: pd.DataFrame):
    """
    Top toolbar: Market Cap slider (safe), Sector multiselect, Industry multiselect, Reset.
    Applies to companyData FIRST; sections then merge onto that 'universe'.
    """
    cdf = company_df.copy()

    # Ensure required columns
    for c in ("market_cap", "sector", "industry"):
        if c not in cdf.columns:
            cdf[c] = np.nan

    # Valid market caps
    mc_series = pd.to_numeric(cdf["market_cap"], errors="coerce")
    mc_valid = mc_series.dropna()
    mc_has_range = (not mc_valid.empty) and (float(mc_valid.min()) < float(mc_valid.max()))

    st.markdown('<div class="toolbar">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([0.35, 0.25, 0.25, 0.15])

    # --- Market Cap control (safe) ---
    with col1:
        if mc_has_range:
            mc_min = float(mc_valid.min())
            mc_max = float(mc_valid.max())
            step = max((mc_max - mc_min) / 100, 1.0)
            mc_val = st.slider(
                "Market Cap (USD)",
                min_value=mc_min,
                max_value=mc_max,
                value=(mc_min, mc_max),
                step=step,
                key="uni_mc",
            )
        else:
            mc_val = None  # filter disabled
            st.write("Market Cap (USD)")
            st.caption("No valid range detected → filter off")

    # --- Sector / Industry ---
    with col2:
        sectors = sorted([s for s in cdf["sector"].dropna().unique().tolist()])
        sec_val = st.multiselect("Sector", sectors, default=sectors, key="uni_sector")
    with col3:
        inds = sorted([s for s in cdf["industry"].dropna().unique().tolist()])
        ind_val = st.multiselect("Industry", inds, default=inds, key="uni_ind")

    # --- Reset ---
    with col4:
        if st.button("Reset Filters", key="uni_reset"):
            for k in list(st.session_state.keys()):
                if k.startswith("sl_") or k in ("uni_mc", "uni_sector", "uni_ind"):
                    del st.session_state[k]
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Apply the universe filters
    out = cdf.copy()
    if mc_val is not None:
        out = out[mc_series.between(mc_val[0], mc_val[1])]
    if sec_val:
        out = out[out["sector"].isin(sec_val)]
    if ind_val:
        out = out[out["industry"].isin(ind_val)]
    return out

def metric_sliders(df: pd.DataFrame, section: str):
    """
    Auto-build sliders for ALL numeric metrics in the current section.
    Uses 5th–95th percentile as default range.
    """
    if df.empty:
        return df
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("market_cap",)]
    if not num_cols:
        return df

    with st.expander("🔎 Refine by metrics", expanded=True):
        cols = st.columns(3)
        mask = pd.Series(True, index=df.index)
        for i, c in enumerate(num_cols):
            s = pd.to_numeric(df[c], errors="coerce")
            s_valid = s.dropna()
            if s_valid.empty:
                continue
            vmin = float(s_valid.min()); vmax = float(s_valid.max())
            if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
                continue
            lo_q = float(s_valid.quantile(0.05)); hi_q = float(s_valid.quantile(0.95))
            step = (hi_q - lo_q) / 100.0 if hi_q > lo_q else max((vmax - vmin) / 100.0, 1e-6)
            val = cols[i % 3].slider(
                c, min_value=vmin, max_value=vmax,
                value=(lo_q, hi_q), step=step, key=f"sl_{section}_{c}"
            )
            mask &= s.between(val[0], val[1])
        df = df[mask]
    return df

def export_buttons(df: pd.DataFrame, filename: str = "screener_export"):
    c1, c2 = st.columns([0.85, 0.15])
    with c2:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export CSV",
            data=csv,
            file_name=f"{filename}.csv",
            mime="text/csv",
            key=f"dl_{filename}",
        )