# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import altair as alt
from utils.config import dico_sectors

def sectors(weights_ptf,asset_names):
    # Create reverse mapping: company → sector
    company_to_sector = {
        company: sector
        for sector, company_list in dico_sectors.items()
        for company in company_list
    }

    # Build the list of sectors in the same order as 'companies'
    sectors_in_order = [company_to_sector[company] for company in asset_names]
    st.subheader("Portfolio Repartition in sectors")
    fig2, ax2 = plt.subplots(figsize=(8, 10))

    threshold = 0.002
    filtered_weights = []
    filtered_sectors = []
    for w, a in zip(weights_ptf, sectors_in_order):
        if w > threshold:
                filtered_weights.append(w)
                filtered_sectors.append(a)

    chart_data = pd.DataFrame({"Sector": filtered_sectors, "Weight": filtered_weights})
        
    chart = (
    alt.Chart(chart_data)
    .mark_arc(outerRadius=120, innerRadius=60)
    .encode(
    theta="Weight",
    color=alt.Color("Sector", legend=alt.Legend(title="Sectors repartition"), scale=alt.Scale(scheme="blues")),
    tooltip=["Sector", alt.Tooltip("Weight", format=".2%")],))

    st.altair_chart(chart, use_container_width=True)


def weights_tabledisplay(weights_ptf,assets_names_ptf):
    df_table_R = pd.DataFrame({"Repartition": [f"{w:.2%}" for w in weights_ptf]}, index=assets_names_ptf)
    df_table_R.index.name = "Stock"

    html = df_table_R.T.to_html(index=True, justify="center", border=0, classes="styled-table")

    st.markdown(
        """
    <style>
    .nice-table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 14px; font-family: Arial, sans-serif; text-align: center; }
    .nice-table th, .nice-table td { border: 1px solid #d0d0d0; padding: 8px 10px; }
    .nice-table th { background-color: #f5f5f5; color: #222; font-weight: 600; }
    .nice-table tr:nth-child(even) { background-color: #fafafa; }
    </style>
    """,
        unsafe_allow_html=True,)
    st.subheader("Portfolio Repartition")
    st.markdown(html, unsafe_allow_html=True)

def performance_graph(cum_perf,horizon):
        df = cum_perf.reset_index()
        df.columns = ["Date", "Value"]


        delta = round(df["Value"].iloc[-1] - df["Value"].iloc[1],2)
        move=round(100*delta/df["Value"].iloc[1],2)

        if delta > 0:
            chart_subtitle = "+ " + str(delta) + " € in " + horizon + " : +" + str(((move))) + " %"
            color = "green"
        else:
            chart_subtitle = str((delta)) + " € in " + horizon + " : " + str(((move))) + " %"
            color = "red"

        chart_title="Historical Portfolio's price evolution"
        st.subheader(chart_title)
        y_min = round(df["Value"].min() * 0.95,0)  # 5% below the actual min
        y_max = round(df["Value"].max() * 1.05,0)  # 5% above the max (optional padding)
        y_axis = alt.Y("Value:Q", title="Value", scale=alt.Scale(domain=[y_min, y_max],zero=False))

        
        # Invisible rule just to force axis
        axis_rule = alt.Chart(df).mark_rule(opacity=0).encode(
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[y_min, y_max], zero=False))
        )

        line = alt.Chart(df).mark_line(color=color).encode(
            x="Date:T",
            y="Value:Q",
            tooltip=["Date:T", alt.Tooltip("Value:Q", format=".2f")]
        )
        # Area from a baseline (like y_min) up to the line
        area = alt.Chart(df).mark_area(opacity=0.15, color=color).encode(
            x="Date:T",
            y="Value:Q",
            y2=alt.Y2Value(y_min))  # baseline for the area
        
        chart = (axis_rule+line).properties(
            title=alt.TitleParams(
                text=chart_subtitle,
                color=color,
                anchor="middle" ))
        #area = chart.mark_area(opacity=0.15)
        st.altair_chart(chart, use_container_width=True)

def weights_graph(weights_ptf,asset_names):
        st.subheader("Portfolio Repartition")
        fig2, ax2 = plt.subplots(figsize=(8, 10))

        threshold = 0.002
        filtered_weights = []
        filtered_assets = []
        for w, a in zip(weights_ptf, asset_names):
            if w > threshold:
                filtered_weights.append(w)
                filtered_assets.append(a)

        chart_data = pd.DataFrame({"Stock": filtered_assets, "Weight": filtered_weights})
        
        chart = (
        alt.Chart(chart_data)
        .mark_arc(outerRadius=120, innerRadius=60)
        .encode(
            theta="Weight",
            color=alt.Color("Stock", legend=alt.Legend(title="Holdings"), scale=alt.Scale(scheme="purples")),
            tooltip=["Stock", alt.Tooltip("Weight", format=".2%")],))

        st.altair_chart(chart, use_container_width=True)

def Portfolio_presentation(type_name, weights, assets_names, used_returns, used_px):
    today = date.today()

    portfolio_returns = used_returns.dot(weights)
    portfolio_px = used_px.dot(weights)

    portfolio_annualized_return = np.mean(portfolio_returns) * 252
    portfolio_annualized_vol = np.std(portfolio_returns) * np.sqrt(252)
    portfolio_annualized_sharpe=0.0 if portfolio_annualized_vol == 0 else (portfolio_annualized_return / portfolio_annualized_vol)
    
    st.title(type_name + " Portfolio")
    st.write("")

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Expected annual return", f"{portfolio_annualized_return*100:.2f}%")
    with metric_cols[1]:
        st.metric("Expected annual volatility", f"{portfolio_annualized_vol*100:.2f}%")
    with metric_cols[2]:
        st.metric("Return / Risk score", f"{portfolio_annualized_sharpe:.2f}")

    # -------------------------------
    # Table of weights
    weights_tabledisplay(weights,assets_names)

    # Horizon selection and data filtration
    periods = ["1M", "6M", "1Y", "5Y", "10Y", "20Y", "MAX"]
    horizon = st.radio("**Choose horizon:**", periods, horizontal=True)

    if horizon != "MAX":
        if horizon[-1] == "Y":
            target = date.today() - relativedelta(years=int(horizon[:-1]))
        elif horizon[-1] == "M":
            target = date.today() - relativedelta(months=int(horizon[:-1]))
        closest_idx = portfolio_px.index.get_indexer([target], method="nearest")[0]
        closest_date = portfolio_px.index[closest_idx]
        filtrated_data = portfolio_px[portfolio_px.index >= closest_date]
    else:
        filtrated_data = portfolio_px

    # Plots
    col1, col2 = st.columns(2)

    with col1:
        performance_graph(filtrated_data,horizon)
        
    with col2:
        weights_graph(weights,assets_names)
    
    col1, col2,col3 = st.columns([1, 2, 1])

    with col2:
        sectors(weights,assets_names)
        
