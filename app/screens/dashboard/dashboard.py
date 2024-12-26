import streamlit as st
from utils.session_manager_utils import logout
from .investments_table import update_investments_table


def dashboard_screen():
    st.sidebar.markdown("## *Powered by FischerAI*")
    st.sidebar.title("Menu")
    st.sidebar.button("Log out", on_click=logout)

    st.title("Welcome to the RiskMaster!")

    table_investments_placeholder = st.empty()

    update_investments_table(table_investments_placeholder)
