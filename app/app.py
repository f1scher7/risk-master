import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import streamlit as st
from enums import Screen
from utils.session_manager_utils import initialize_session
from screens.login_screen import login_screen
from screens.sidebar import generate_sidebar
from screens.dashboard.dashboard import dashboard_screen
from screens.price_prediction_screen import price_prediction_screen
from screens.risk_analysis_screen import risk_analysis_screen


st.set_page_config(
    page_title="RiskMaster",
)


def risk_master():
    initialize_session()

    if not st.session_state["is_logged_in"]:
        login_screen()
    else:
        current_screen = st.session_state["current_screen"]

        generate_sidebar(current_screen == Screen.DASHBOARD)

        if current_screen == Screen.DASHBOARD:
            st.empty()
            dashboard_screen()
        elif current_screen == Screen.PRICE_PREDICTION:
            st.empty()
            price_prediction_screen()
        elif current_screen == Screen.RISK_ANALYSIS:
            st.empty()
            risk_analysis_screen()


if __name__ == "__main__":
    risk_master()