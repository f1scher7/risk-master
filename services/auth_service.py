from bcrypt import checkpw
from uuid import uuid4
from services.user_service import UserService


class AuthService:
    @staticmethod
    def login(username: str, password: str):
        user = UserService.get_user_by_username(username);

        if user and AuthService.is_password_correct(password=password ,hashed_password=user.password):
            user.session_token = str(uuid4())

            UserService.update_user(updated_user=user)

            return {"SUCCESS": f"User {user.username} logged in", "username": user.username, "session_token": f"{user.session_token}"}

        return {"ERROR": "Invalid username or password"}

    @staticmethod
    def logout(username: str):
        user = UserService.get_user_by_username(username)

        if user and user.session_token is not None:
            user.session_token = None

            UserService.update_user(updated_user=user)

            return {"SUCCESS": f"User {user.username} legged out"}

        return {"ERROR": f"User {user.username} not found"}



    @staticmethod
    def is_password_correct(password, hashed_password):
        return checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))