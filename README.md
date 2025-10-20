# Portfolio Optimization App

A Streamlit application for selecting and optimizing European stock portfolios using historical data.

## Features
- Extracts index components from Wikipedia (CAC 40, DAX, etc.)
- Filters stocks with at least 20 years of historical data
- Downloads adjusted prices from Yahoo Finance
- Optimizes portfolio using:
  - Minimum Volatility
  - Maximum Return
  - Risk-Optimized Minimum Volatility
- Displays portfolio weights visually

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
