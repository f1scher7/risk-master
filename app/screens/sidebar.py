import streamlit as st
from enums import Screen
from utils.session_manager_utils import logout
from utils.app_utils import set_current_screen


def generate_sidebar(is_dashboard):
    st.sidebar.markdown("## *Powered by FischerAI*")

    st.sidebar.title("Menu")

    if st.sidebar.button("Price Prediction", use_container_width=True):
        set_current_screen(Screen.PRICE_PREDICTION)

    if st.sidebar.button("Risk Analysis", use_container_width=True):
        set_current_screen(Screen.RISK_ANALYSIS)

    if not is_dashboard:
        if st.sidebar.button("Back to dashboard", use_container_width=True):
            set_current_screen(Screen.DASHBOARD)

    st.sidebar.button("Log out", type="primary", on_click=logout)
