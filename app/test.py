# import streamlit as st
# from dateutil.utils import today
#
# st.title("Risk-Master")
# st.sidebar.title("Parameters")
#
# end_date = st.sidebar.date_input("End date", min_value=today())
#
# st.write("asd")

from services.user_service import UserService


UserService.create_user("tempuser", "123")