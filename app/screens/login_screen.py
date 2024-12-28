import streamlit as st
from enums import Screen
from services.auth_service import AuthService
from utils.app_utils import set_current_screen


def login_screen():
    st.title("Log in")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        response = AuthService.login(username, password)

        if "SUCCESS" in response:
            st.session_state["is_logged_in"] = True
            st.session_state["username"] = response["username"]
            st.session_state["session_token"] = response["session_token"]

            set_current_screen(Screen.DASHBOARD)

            st.rerun()
        else:
            st.error(response["ERROR"])
