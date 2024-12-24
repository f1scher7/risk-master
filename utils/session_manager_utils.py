import streamlit as st
from services.auth_service import AuthService


def initialize_session():
    if "is_logged_in" not in st.session_state:
        st.session_state["is_logged_in"] = False
        st.session_state["username"] = None
        st.session_state["session_token"] = None


def logout():
    AuthService.logout(st.session_state["username"])

    st.session_state["is_logged_in"] = False
    st.session_state["username"] = None
    st.session_state["session_token"] = None

    st.rerun()