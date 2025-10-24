
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
    
