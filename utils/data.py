# utils/data.py
from pathlib import Path
import pandas as pd
import streamlit as st

def _first_existing(paths):
    for p in map(Path, paths):
        if p.exists():
            return p
    return None

def _read_any(p: Path | None) -> pd.DataFrame:
    if not p:
        return pd.DataFrame()
    try:
        if p.suffix.lower() in (".parquet",):
            return pd.read_parquet(p)
        if p.suffix.lower() in (".xls", ".xlsx"):
            return pd.read_excel(p)
        # fallback
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_all(data_dir: str | Path = "data_private") -> dict[str, pd.DataFrame]:
    d = Path(data_dir)

    # --- company universe (your repo has companyData_names.parquet) ---
    company_file = _first_existing([
        d / "companyData.parquet",
        d / "companyData_names.parquet",
        d / "companyData.xlsx",  # optional fallback
    ])
    company_df = _read_any(company_file)

    # normalize column names if needed
    if not company_df.empty:
        ren = {}
        if "Ticker" in company_df.columns and "symbol" not in company_df.columns:
            ren["Ticker"] = "symbol"
        if "Company" in company_df.columns and "companyName" not in company_df.columns:
            ren["Company"] = "companyName"
        if ren:
            company_df = company_df.rename(columns=ren)

    # --- market cap wide (your repo has mktCapGBP.parquet) ---
    mcap_file = _first_existing([
        d / "marketCapGBP.parquet",
        d / "mktCapGBP.parquet",
    ])
    mcap_wide = _read_any(mcap_file)

    return {
        "companyData": company_df,
        "marketCapGBP": mcap_wide,
    }

# Optional helpers you had; keep them if other modules import them
def load_from_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

def merge_market_caps(company_df: pd.DataFrame, mktcap_df: pd.DataFrame):
    if company_df.empty or mktcap_df.empty:
        return company_df
    mktcap_df = mktcap_df.copy()
    if "ticker" in mktcap_df.columns:
        mktcap_df["ticker"] = mktcap_df["ticker"].astype(str).str.upper().str.strip()
    merged = company_df.merge(
        mktcap_df[["ticker", "market_cap_gbp"]],
        on="ticker", how="left"
    )
    return merged

def coerce_identifier_cols(df: pd.DataFrame, ticker_col: str = "ticker"):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if ticker_col in df.columns:
        df[ticker_col] = df[ticker_col].astype(str).str.upper().str.strip()
    if "isin" in df.columns:
        df["isin"] = df["isin"].astype(str).str.strip()
    for c in ("market_cap", "price"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("sector", "industry", "country", "currency"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df