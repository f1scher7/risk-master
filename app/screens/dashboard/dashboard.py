import streamlit as st
from .investments_table import update_investments_table


def dashboard_screen():
    st.title("Welcome to the RiskMaster!")

    table_investments_placeholder = st.empty()

    update_investments_table(table_investments_placeholder)
