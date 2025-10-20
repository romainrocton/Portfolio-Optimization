# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd


def weights_plots_minv(type_name, weights, assets_names, used_returns, used_px):
    today = date.today()

    portfolio_returns = used_returns.dot(weights)
    portfolio_px = used_px.dot(weights)

    portfolio_annualized_return = np.mean(portfolio_returns) * 252
    portfolio_annualized_vol = np.std(portfolio_returns) * np.sqrt(252)

    st.title(type_name + " Portfolio")
    st.write("")
    st.write("Portfolio's annualized return : " + str(round(portfolio_annualized_return * 100, 2)) + "%")
    st.write("Portfolio's annualized risk : " + str(round(portfolio_annualized_vol * 100, 2)) + "%")

    # -------------------------------
    # Table of weights
    df_table_R = pd.DataFrame({"Repartition": [f"{w:.1%}" for w in weights]}, index=assets_names)
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
        unsafe_allow_html=True,
    )

    st.subheader("Portfolio Repartition")
    st.markdown(html, unsafe_allow_html=True)

    # Horizon selection
    periods = ["1M", "6M", "1Y", "5Y", "10Y", "20Y", "MAX"]
    horizon = st.radio("Choose horizon:", periods, horizontal=True)

    if horizon != "MAX":
        if horizon[-1] == "Y":
            target = date.today() - relativedelta(years=int(horizon[:-1]))
        elif horizon[-1] == "M":
            target = date.today() - relativedelta(months=int(horizon[:-1]))
        closest_idx = portfolio_px.index.get_indexer([target], method="nearest")[0]
        closest_date = portfolio_px.index[closest_idx]
        cum_perf = portfolio_px[portfolio_px.index >= closest_date]
    else:
        cum_perf = portfolio_px

    # Plots
    col1, col2 = st.columns(2)
    sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.2)

    with col1:
        curr = cum_perf[-1]
        fst = cum_perf[1] if len(cum_perf) > 1 else cum_perf[0]
        delta = curr - fst
        if delta > 0:
            sub = "+ " + str(round(delta, 2)) + " € in " + horizon + " : +" + str(round((delta / fst) * 100, 2)) + " %"
            color = "green"
            cl = "Greens"
        else:
            sub = str(round(delta, 2)) + " € in " + horizon + " : " + str(round((delta / fst) * 100, 2)) + " %"
            color = "red"
            cl = "Reds"

        st.subheader("Historical Portfolio's price evolution")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=cum_perf.index, y=cum_perf.values, ax=ax1, linewidth=2)

        fig1.suptitle("Portfolio : " + str(round(curr, 2)) + " € ", fontsize=15, fontweight="bold", ha="center")
        ax1.set_title(sub, fontsize=12, color=color, fontstyle="italic", pad=15, ha="center")

        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xlabel("Date", fontsize=10, fontstyle="italic", labelpad=10)
        ax1.set_ylabel("Price", fontsize=10, fontstyle="italic", labelpad=10)
        if (horizon == "20Y") or (horizon == "10Y") or (horizon == "MAX"):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        elif horizon == "1M":
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))

        locator = mdates.AutoDateLocator(maxticks=6)
        ax1.xaxis.set_major_locator(locator)
        ax1.grid(False)
        ax1.legend(loc="lower right", frameon=False)
        st.pyplot(fig1)

    with col2:
        st.subheader("Portfolio Repartition")
        fig2, ax2 = plt.subplots(figsize=(8, 10))

        threshold = 0.002
        filtered_weights = []
        filtered_assets = []
        for w, a in zip(weights, assets_names):
            if w > threshold:
                filtered_weights.append(w)
                filtered_assets.append(a)

        # Use seaborn palette but do not force colors
        wedges, texts, autotexts = ax2.pie(filtered_weights, labels=None, autopct="%1.1f%%", startangle=90)
        ax2.legend(wedges, filtered_assets, title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        fig2.gca().add_artist(centre_circle)
        st.pyplot(fig2)
