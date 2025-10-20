# utils/data_extraction.py
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta


def extract_index_components(wiki_url, index_name):
    headers = {"User-Agent": "PortfolioApp/1.0 (contact@example.com)"}
    response = requests.get(wiki_url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(response.text)

    for table in tables:
        # detect candidate columns in many wiki tables
        if 'Company' in table.columns or 'Security' in table.columns or 'Name' in table.columns:
            companies = []
            for _, row in table.iterrows():
                company_info = {
                    "company": row.get("Company") or row.get("Security") or row.get("Name"),
                    "ticker": row.get("Ticker") or row.get("Symbol") or row.get("Ticker symbol") or None
                }
                companies.append(company_info)
            return companies
    return None


@st.cache_data(ttl=86400)
def extract_corpos(indices):
    """
    returns (tickers, t, corpo) similar to original notebook:
      - tickers: list of unique tickers
      - t: concatenated tickers (possibly with suffixes)
      - corpo: concatenated company names
    """
    cac40_name = dax40_name = aex25_name = ibex35_name = ftsemib_name = []
    cac40_ticker = dax40_ticker = aex25_ticker = ibex35_ticker = ftsemib_ticker = []

    all_tickers = []
    all_names = []
    t = []
    for index_name, url in indices.items():
        companies = extract_index_components(url, index_name)
        if not companies:
            continue
        names = [c["company"] for c in companies]
        tickers = [c["ticker"] for c in companies]
        if index_name == "CAC 40":
            cac40_name, cac40_ticker = names, tickers
        elif index_name == "DAX 40":
            dax40_name, dax40_ticker = names, tickers
        elif index_name == "AEX 25":
            aex25_name, aex25_ticker = names, tickers
        elif index_name == "IBEX 35":
            ibex35_name, ibex35_ticker = names, tickers
        elif index_name == "FTSE MIB":
            ftsemib_name, ftsemib_ticker = names, tickers

        all_tickers += tickers
        all_names += names

    # For AEX tickers add .AS suffix if needed (original logic)
    try:
        for i in range(len(aex25_ticker)):
            if aex25_ticker[i] and not aex25_ticker[i].endswith(".AS"):
                aex25_ticker[i] = aex25_ticker[i] + ".AS"
    except Exception:
        pass

    t = cac40_ticker + dax40_ticker + aex25_ticker + ibex35_ticker + ftsemib_ticker
    tickers = list(dict.fromkeys(t))  # unique preserving order
    c = cac40_name + dax40_name + aex25_name + ibex35_name + ftsemib_name
    corpo = list(dict.fromkeys(c))
    return tickers, t, corpo


@st.cache_data(ttl=86400)
def old_tickers(years, tick):
    """Keep only tickers that have first date <= years ago"""
    today = date.today()
    cutoff = today - relativedelta(years=years)
    valid_tickers = []
    for t in tick:
        try:
            df = yf.download(t, period="max", progress=False, auto_adjust=True)
            if df.empty:
                continue
            first_date = df.index.min().date()
            if first_date <= cutoff:
                valid_tickers.append(t)
        except Exception:
            continue
    return valid_tickers


@st.cache_data(ttl=86400)
def EUR_tickers(tick):
    good = []
    for t in tick:
        try:
            info = yf.Ticker(t).info
            if info.get("currency") == "EUR":
                good.append(t)
        except Exception:
            continue
    return good


@st.cache_data(ttl=86400)
def extract_data(valid_tickers):
    if not valid_tickers:
        st.warning("No valid tickers.")
        return None

    today = date.today()
    thirty_years_ago = today - relativedelta(years=30)
    data = pd.DataFrame()

    for ti in valid_tickers:
        try:
            df = yf.download(ti, start=thirty_years_ago, end=today, progress=False, auto_adjust=True)["Close"]
            df.name = ti
            data = pd.concat([data, df], axis=1)
        except Exception:
            continue

    return data
