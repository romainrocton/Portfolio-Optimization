
# theme.py
from utils.config import THEME_COLORS
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import base64

def add_logo(image_path:str):

    with open(image_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    st.markdown(
            f"""
            <style>
                .logo-container {{
                    position: absolute;
                    top: 15px;
                    right: 25px;
                    width: 450px;
                    z-index: 999;
                }}
                /* Enlève la marge créée par Streamlit */
                section[data-testid="stSidebar"] + div {{
                    margin-top: -60px;
                }}
            </style>
            <div class="logo-container">
                <img src="data:image/png;base64,{encoded}">
            </div>
            """,
            unsafe_allow_html=True
        )
    

def apply_theme():
    """Apply the app's theme to Streamlit UI, Matplotlib, and Plotly charts."""

    # --- Streamlit CSS ---
    st.markdown(
        f"""
        <style>
        /* App background and text */
        .stApp {{
            background-color: {THEME_COLORS['section_bg']};
            color: {THEME_COLORS['text_primary']};
        }}

        /* Headings */
        h1, h2, h3, h4 {{
            color: {THEME_COLORS['primary']};
        }}

        /* Buttons */
        .stButton>button {{
            background-color: {THEME_COLORS['slider_active']};
            color: {THEME_COLORS['tag_text']};
        }}

        /* Metrics */
        .stMetric > div {{
            background-color: {THEME_COLORS['metric_bg']};
        }}

        /* Optional: Sliders */
        .stSlider .css-1aumxhk .stSlider>div>div>div {{
            background-color: {THEME_COLORS['slider_active']};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Matplotlib ---
    #plt.rcParams['axes.facecolor'] = THEME_COLORS['section_bg']
    #plt.rcParams['axes.edgecolor'] = THEME_COLORS['border']
    #plt.rcParams['figure.facecolor'] = THEME_COLORS['section_bg']
    #plt.rcParams['text.color'] = THEME_COLORS['text_primary']
    #plt.rcParams['xtick.color'] = THEME_COLORS['text_secondary']
    #plt.rcParams['ytick.color'] = THEME_COLORS['text_secondary']

    # --- Plotly ---
    px.colors.qualitative.THEME = [
        THEME_COLORS['primary'],
        THEME_COLORS['accent'],
        THEME_COLORS['slider_active'],
        THEME_COLORS['tag_bg']
    ]
