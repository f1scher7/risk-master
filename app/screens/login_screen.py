import streamlit as st
from services.auth_service import AuthService


def login_screen():
    st.title("Log in")

    username = st.text_input("Username")
    password = st.text_input("Password")

    print(username)
    print(password)

    if st.button("Log in"):
        response = AuthService.login(username, password)

        if "SUCCESS" in response:
            st.success(response["SUCCESS"])
            st.session_state["is_logged_in"] = True
            st.session_state["username"] = response["username"]
            st.session_state["session_token"] = response["session_token"]

            st.rerun()
        else:
            st.error(response["ERROR"])
