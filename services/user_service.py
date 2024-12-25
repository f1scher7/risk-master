from models.user import User
from utils.db_connection_utils import get_session
from sqlalchemy.exc import IntegrityError
from bcrypt import hashpw
from bcrypt import gensalt

class UserService:

    @staticmethod
    def create_user(username: str, password: str):
        try:
            hashed_password = hashpw(password.encode('utf-8'), gensalt()).decode('utf-8')

            user = User(username=username, password=hashed_password)

            with get_session() as db_session:
                db_session.add(user)
                db_session.commit()

            return user
        except IntegrityError:
            return {"ERROR": "Username already exists"}


    @staticmethod
    def get_user_by_id(user_id: int):
        with get_session() as db_session:
            user = db_session.query(User).filter(User.id == user_id).first()
        return user


    @staticmethod
    def get_user_by_username(username: str):
        with get_session() as db_session:
            user = db_session.query(User).filter(User.username == username).first()
        return user


    @staticmethod
    def get_all_users():
        with get_session() as db_session:
            users = db_session.query(User).all()
        return users


    @staticmethod
    def update_user(updated_user: User):
        with get_session() as db_session:
            user = UserService.get_user_by_id(updated_user.id)

            if user:
                user.username = updated_user.username
                user.session_token = updated_user.session_token

                db_session.commit()

                return {"SUCCESS": f"User {user.username} was updated"}
            else:
                return {"ERROR": f"User doesnt exist"}


    @staticmethod
    def delete_user_by_username(username: str):
        user = UserService.get_user_by_username(username)

        if user:
            with get_session() as db_session:
                db_session.delete(user)
                db_session.commit()
            return {"SUCCESS": f"User {username} was deleted"}
        else:
            print(f'User {username} not found')
