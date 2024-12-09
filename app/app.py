import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from services.auth_service import AuthService


if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False
    st.session_state["username"] = None
    st.session_state["session_token"] = None


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
            st.experimental_rerun()
        else:
            st.error(response["ERROR"])

def logout():
    AuthService.logout(st.session_state["username"])

    st.session_state["is_logged_in"] = False
    st.session_state["username"] = None
    st.session_state["session_token"] = None
    st.experimental_rerun()

def main():
    if not st.session_state["is_logged_in"]:
        login_screen()
    else:
        st.sidebar.title("Menu")
        st.sidebar.button("Log out", on_click=logout)

        st.title("Witaj w aplikacji Risk-Master!")
        st.write(f"Zalogowany jako: {st.session_state['username']}")

if __name__ == "__main__":
    main()