import base64

import streamlit as st
from env_loader import STYLES_PATH


def load_css(file_name: str):
    file_path = f"{STYLES_PATH}{file_name}"

    with open(file_path) as f:
        css = f.read()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def encode_image_to_base64(image_path):
    return base64.b64encode(open(image_path, "rb").read()).decode()
