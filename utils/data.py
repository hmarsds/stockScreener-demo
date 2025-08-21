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
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        if p.suffix.lower() in (".xls", ".xlsx"):
            return pd.read_excel(p)
        return pd.read_parquet(p)  # fallback
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_all(data_dir: str | Path = "data_private") -> dict[str, pd.DataFrame]:
    d = Path(data_dir)

    # ---------- Company universe ----------
    # Accept either the “names map” (tickers in index, 1 column) or the Excel file
    names_file = _first_existing([
        d / "companyData_names.parquet",
        d / "companyData.parquet",   # if you ever add it later
    ])
    excel_file = _first_existing([d / "companyData.xlsx"])

    names_df = _read_any(names_file)
    excel_df = _read_any(excel_file)

    # Normalize names_df -> columns: symbol, companyName
    company_df = pd.DataFrame()
    if not names_df.empty:
        # If 1 column with tickers in index (your format)
        if names_df.shape[1] == 1:
            only_col = names_df.columns[0]
            tmp = names_df.reset_index()
            # Typical index name is 'Ticker'; use whatever it is
            idx_col = tmp.columns[0]
            tmp = tmp.rename(columns={idx_col: "symbol", only_col: "companyName"})
            company_df = tmp
        else:
            # If it already has reasonable columns, try to map them
            tmp = names_df.copy()
            ren = {}
            if "Ticker" in tmp.columns and "symbol" not in tmp.columns:
                ren["Ticker"] = "symbol"
            if "Company" in tmp.columns and "companyName" not in tmp.columns:
                ren["Company"] = "companyName"
            tmp = tmp.rename(columns=ren)
            if "symbol" not in tmp.columns and tmp.index.name:
                tmp = tmp.reset_index().rename(columns={tmp.columns[0]: "symbol"})
            if "companyName" not in tmp.columns:
                # Try a best-effort guess
                for cand in ("name", "Name", "company", "Company"):
                    if cand in tmp.columns:
                        tmp = tmp.rename(columns={cand: "companyName"})
                        break
            company_df = tmp

    # Optionally merge richer attributes from Excel (if present)
    if not excel_df.empty:
        # Excel already has: symbol, price, marketCap, companyName, currency, isin, exchangeFullName, exchange, sector, industry, ceo, country
        # Just ensure the key columns exist and are clean
        ex = excel_df.copy()
        # Normalize potential header variations
        ex = ex.rename(columns={
            "Ticker": "symbol",
            "Company": "companyName",
        })
        if "symbol" in ex.columns:
            ex["symbol"] = ex["symbol"].astype(str).str.upper().str.strip()
        if "companyName" in ex.columns:
            ex["companyName"] = ex["companyName"].astype(str).str.strip()
        # If we had no names_df, use excel only; else outer-merge to enrich
        if company_df.empty:
            company_df = ex
        else:
            company_df["symbol"] = company_df["symbol"].astype(str).str.upper().str.strip()
            company_df = company_df.merge(
                ex, on="symbol", how="left", suffixes=("", "_ex")
            )
            # Prefer existing companyName; fill from excel if missing
            if "companyName_ex" in company_df.columns:
                company_df["companyName"] = company_df["companyName"].fillna(company_df["companyName_ex"])
                company_df = company_df.drop(columns=[c for c in company_df.columns if c.endswith("_ex")])

    # Final cleanups & required columns
    if not company_df.empty:
        for c in ("symbol", "companyName"):
            if c not in company_df.columns:
                company_df[c] = pd.NA
        company_df["symbol"] = company_df["symbol"].astype(str).str.upper().str.strip()
        company_df["companyName"] = company_df["companyName"].astype(str).str.strip()
        company_df = company_df.dropna(subset=["symbol"]).drop_duplicates("symbol")

    # ---------- Market cap wide (GBP) ----------
    mcap_file = _first_existing([
        d / "mktCapGBP.parquet",
        d / "marketCapGBP.parquet",
    ])
    mcap_wide = _read_any(mcap_file)

    return {
        "companyData": company_df,
        "marketCapGBP": mcap_wide,
    }

# Optional helpers kept for imports elsewhere
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