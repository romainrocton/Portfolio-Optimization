# app.py
import streamlit as st
import numpy as np
from PIL import Image



from utils.data_extraction import (
    extract_index_components,
    extract_corpos,
    EUR_tickers,
    old_tickers,
    extract_data,
)
from utils.optimization import port_minvol, port_maxret, port_minvol_ro
from utils.simulation import simul_EF
from utils.visualization import weights_plots_minv
from utils.config import THEME_COLORS
from utils.theme import add_logo, apply_theme

# --- Background & Texte 
#apply_theme()

st.set_page_config(page_title="European stocks selector & optimizer", layout="wide",initial_sidebar_state="collapsed")
# --- LOGO FIXE EN HAUT À DROITE ---
image_path = "logo/Quantiva.PNG"
add_logo(image_path)



def main():
    st.title("Portfolio Optimization")

    indices = {
        "CAC 40": "https://en.wikipedia.org/wiki/CAC_40",
        "DAX 40": "https://en.wikipedia.org/wiki/DAX",
        "AEX 25": "https://en.wikipedia.org/wiki/AEX_index",
        "IBEX 35": "https://en.wikipedia.org/wiki/IBEX_35",
        "FTSE MIB": "https://en.wikipedia.org/wiki/FTSE_MIB",
    }

    # Extract components
    tickers, tick, corpo = extract_corpos(indices)

    # Filter tickers with at least 20 years of data and quoted in EUR
    valid_tickers = EUR_tickers(old_tickers(20, tickers))

    # Download historical data for valid tickers
    data = extract_data(valid_tickers)

    if data is None or data.empty:
        st.warning("No data available yet (check connectivity or tickers).")
        return

    # Build index name lists (names and tickers per index) for selection UI
    # We'll re-extract to get per-index names as in your original
    index_to_actions = {}
    for index_name, url in indices.items():
        companies = extract_index_components(url, index_name)
        if not companies:
            index_to_actions[index_name] = []
            continue
        names = [c["company"] for c in companies]
        index_to_actions[index_name] = names


    # map ticker -> company name
    dico_index = {}
    for t, name in zip(tick, corpo):
        if t not in dico_index:
            dico_index[t] = name

    for index_name, names in index_to_actions.items():
        index_to_actions[index_name] = [n for n in names if n in dico_index.values()]
    

    # rename columns to company names where possible and drop duplicates
    data = data.rename(columns=dico_index)
    data = data.loc[:, ~data.columns.duplicated()]
    data_c = data.dropna(axis=0)
    data_c.sort_index(inplace=True)
    returns = data_c.pct_change().dropna()

    # UI: stock selection
    st.write("")
    selected_display = st.empty()
    st.write("")
    selected_companies = []
    for index_name, actions in index_to_actions.items():
        with st.expander(index_name):
            selected = st.multiselect(f"Select stocks from {index_name}", options=actions, key=f"ms_{index_name}")
            selected_companies.extend(selected)

    if selected_companies:
        selected_display.markdown(
            "<h4 style='margin-bottom: 10px;'>Selected Stocks:</h4>" +
            "".join([
                f"<span style='font-size:15px; background-color:#3a3f51; "
                f"color:#f8f9fa; border-radius:10px; padding:5px 10px; "
                f"margin:3px; display:inline-block;'>{item}</span>"
                for item in selected_companies
            ]),
            unsafe_allow_html=True
        )
    else:
        selected_display.markdown("No stocks selected yet.")
    selected = selected_companies
    
    # Session state initialization
    if "validated" not in st.session_state:
        st.session_state.validated = False
    if "submitt" not in st.session_state:
        st.session_state.submitt = False
    if "typptf" not in st.session_state:
        st.session_state.typptf = "Minimum Risk"
    if "MT" not in st.session_state:
        st.session_state.MT = "Proposition of Portfolios with Inputed Stocks"
    if "chosen_ptf" not in st.session_state:
        st.session_state.chosen_ptf = "High Return/Risk"
    if "weightsorreturn" not in st.session_state:
        st.session_state.weightsorreturn = "Investment Percentage"
    if "weights" not in st.session_state:
        st.session_state.weights = None

    validate = st.button("✅ Validate Selection")
    if validate:
        if len(selected) == 0:
            st.info("Please select at least 1 stock before validation")
            st.session_state.validated = False
        elif len(selected) > 10:
            st.error("❌ You can select up to 10 stocks")
            st.session_state.validated = False
        else:
            st.success(f"✅ You selected {len(selected)} stock(s).")
            st.session_state.validated = True

    # After validation: compute and display
    if st.session_state.validated:
        used_returns = returns.loc[:, returns.columns.isin(selected)]
        used_px = data_c.loc[:, data_c.columns.isin(selected)]

        selected_returns = np.array(used_returns)
        assets_names = used_returns.columns
        nb_assets = len(assets_names)

        mean = np.average(selected_returns, axis=0)
        covariance_matrix = np.cov(selected_returns, rowvar=False)

        # If only 1 stock selected
        if len(selected) == 1:
            w_1stock = np.zeros(len(selected))
            w_1stock[0] = 1
            weights_plots_minv("Single Stock", w_1stock, assets_names, used_returns, used_px)
        else:
            # run simulation to generate a set of candidate portfolios (keeps your original simulation logic)
            weights_MV, weights_S1, weights_S2, weights_S3, weights_MaxRet = simul_EF(
                selected_returns, mean, covariance_matrix, nb_assets
            )

            portfolio_returns_MV = used_returns.dot(weights_MV)
            portfolio_returns_MR = used_returns.dot(weights_MaxRet)
            portfolio_annualized_return_MV = np.mean(portfolio_returns_MV) * 252
            portfolio_annualized_vol_MV = np.std(portfolio_returns_MV) * np.sqrt(252)
            portfolio_annualized_return_MR = np.mean(portfolio_returns_MR) * 252
            portfolio_annualized_vol_MR = np.std(portfolio_returns_MR) * np.sqrt(252)

            PTF_S = ["Low Return/Risk", "Mid Return/Risk", "High Return/Risk"]
            dico_ptf = {
                "Low Return/Risk": weights_S1,
                "Mid Return/Risk": weights_S2,
                "High Return/Risk": weights_S3,
            }
            
            cols = st.columns(3)
            own_vs_guided = ["Proposition of Portfolios with Inputed Stocks", "Investor's choices"]
            for i, management_type in enumerate(own_vs_guided):
                if cols[i].button(management_type):
                    st.session_state.MT = management_type
            MT = st.session_state.MT

            if MT == "Proposition of Portfolios with Inputed Stocks":
                typptf = st.selectbox(
                    "**Choose the portfolio type**",
                    ["Minimum Risk", "Sample"],
                    index=0 if st.session_state.typptf is None else (0 if st.session_state.typptf == "Minimum Risk" else 1),
                )
                st.session_state.typptf = typptf

                if typptf == "Minimum Risk":
                    weights_plots_minv("Minimum Risk", weights_MV, assets_names, used_returns, used_px)
                else:
                    #cols = st.columns(3)
                    risk_level = st.select_slider(
                    "Select your preferred portfolio",
                    options=PTF_S,
                    value=PTF_S[0],  # default selection
                    help="Select your preferred portfolio out of the three options")
                    chosen_ptf = risk_level
                    weights_plots_minv(chosen_ptf, dico_ptf[chosen_ptf], assets_names, used_returns, used_px)
            else:
                invest_choice = ["Investment Percentage", "Target Annual Return"]
                weightsorreturn = st.selectbox(
                    "**What do you wish to choose**", invest_choice, index=0 if st.session_state.weightsorreturn == "Investment Percentage" else 1
                )
                st.session_state.weightsorreturn = weightsorreturn

                if weightsorreturn == "Investment Percentage":
                    with st.form("my_form"):
                        weights_vector = []
                        for item in assets_names:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(item)
                            with col2:
                                w = st.number_input(
                                    f"Weight in {item} (%)", min_value=0.0, max_value=100.0, step=0.01, key=f"input_{item}"
                                )
                                weights_vector.append(w / 100)
                        submitt = st.form_submit_button("✅ Confirm Repartition")
                    if submitt:
                        total_weight = sum(weights_vector)
                        if abs(total_weight - 1) > 1e-6:
                            st.error(f"⚠️ Invalid weights! Total = {total_weight:.2f} (should be 1)")
                        else:
                            st.session_state.weights = np.array(weights_vector)
                    if st.session_state.weights is not None:
                        weights_plots_minv("Your Choice", st.session_state.weights, assets_names, used_returns, used_px)
                else:
                    # target return flow
                    st.markdown("Define your target annualized return")
                    st.write(f"Enter a target between **{portfolio_annualized_return_MV*100:.2f}%** and **{portfolio_annualized_return_MR*100:.2f}%** per year.")
                    target_return = st.number_input(
                        "Objectif de rendement annuel (%)",
                        min_value=portfolio_annualized_return_MV * 100,
                        max_value=portfolio_annualized_return_MR * 100,
                        value=(portfolio_annualized_return_MV * 100 + portfolio_annualized_return_MR * 100) / 2,
                        step=0.1,
                        format="%.2f",
                    )

                    if st.button("Launch Portfolio Optimization"):
                        target_return_daily = target_return / 100 / 252
                        opt_target = port_minvol_ro(mean, covariance_matrix, target_return_daily)
                        st.session_state["opt_target"] = opt_target
                        st.session_state["target_return_daily"] = target_return_daily

                    if "opt_target" in st.session_state:
                        opt_target = st.session_state["opt_target"]
                        weights_plots_minv("Your Optimized", opt_target, assets_names, used_returns, used_px)
                    else:
                        st.info("Choose a target return and click 'Launch Portfolio Optimization' to see the optimized portfolio.")


if __name__ == "__main__":
    main()
