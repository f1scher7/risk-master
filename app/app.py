import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import streamlit as st
from utils.session_manager_utils import initialize_session
from screens.login_screen import login_screen
from screens.dashboard.dashboard import dashboard_screen


st.set_page_config(
    page_title="RiskMaster",
)


def main():
    initialize_session()

    if not st.session_state["is_logged_in"]:
        login_screen()
    else:
        dashboard_screen()


if __name__ == "__main__":
    main()