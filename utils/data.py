import os
import pandas as pd
import streamlit as st

# Private data paths
_DATA_DIR = "/Users/henrymarshall/Desktop/screenerApp/data_private"
_COMPANYDATA_FILE = os.path.join(_DATA_DIR, "companyData.xlsx")
_MKTCAPGBP_FILE = os.path.join(_DATA_DIR, "mktCapGBP.parquet")

@st.cache_data(show_spinner="Loading base datasets…")
def load_all():
    out = {}

    # --- Company Data ---
    if os.path.exists(_COMPANYDATA_FILE):
        df = pd.read_excel(_COMPANYDATA_FILE, sheet_name=0)
        out["companyData"] = _normalize_company_df(df)
    else:
        out["companyData"] = pd.DataFrame(columns=[
            "ticker", "name", "isin", "sector", "industry",
            "country", "currency", "market_cap", "price"
        ])

    # --- Market Cap (GBP) ---
    if os.path.exists(_MKTCAPGBP_FILE):
        out["marketCapGBP"] = load_from_parquet(_MKTCAPGBP_FILE)
    else:
        out["marketCapGBP"] = pd.DataFrame()

    # Placeholders for now
    out["prices"] = pd.DataFrame()
    out["fundamentals_ttm"] = pd.DataFrame()
    out["earnings"] = pd.DataFrame()
    out["estimates"] = pd.DataFrame()
    return out

def load_from_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

def merge_market_caps(company_df: pd.DataFrame, mktcap_df: pd.DataFrame):
    """
    Merge market cap GBP values into companyData based on ticker.
    """
    if company_df.empty or mktcap_df.empty:
        return company_df

    # Ensure tickers match format
    mktcap_df = mktcap_df.copy()
    if "ticker" in mktcap_df.columns:
        mktcap_df["ticker"] = mktcap_df["ticker"].astype(str).str.upper().str.strip()

    merged = company_df.merge(
        mktcap_df[["ticker", "market_cap_gbp"]],
        on="ticker", how="left"
    )
    return merged

def _normalize_company_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure presence of key columns
    for c in ["ticker", "name", "isin", "sector", "industry", 
              "country", "currency", "market_cap", "price"]:
        if c not in df.columns:
            df[c] = pd.NA
    return coerce_identifier_cols(df)

def coerce_identifier_cols(df: pd.DataFrame, ticker_col: str = "ticker"):
    """
    Normalize identifiers/types so filtering & merging are reliable.
    """
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