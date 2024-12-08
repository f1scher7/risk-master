from bcrypt import checkpw
from uuid import uuid4
from db.models import User
from db.db_connection import get_session
from services.user_service import UserService

class AuthService:
    @staticmethod
    def login(username: str, password: str):
        user = UserService.get_user_by_username(username);

        if user and AuthService.is_password_correct(password=password ,hashed_password=user.password):
            user.session_token = str(uuid4())

            UserService.update_user(updated_user=user)

            return {"SUCCESS": f"User {user.username} logged in", "session_token": f"{user.session_token}"}

        return {"ERROR": "Invalid username or password"}

    @staticmethod
    def is_password_correct(password, hashed_password):
        return checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))