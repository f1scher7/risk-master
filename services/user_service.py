from db.models import User
from db.db_connection import get_session
from sqlalchemy.exc import IntegrityError


class UserService:
    @staticmethod
    def create_user(username: str, password: str):
        try:
            user = User(username=username, password=password)

            with get_session() as db_session:
                db_session.add(user)
                db_session.commit()

            return user
        except IntegrityError:
            return {"ERROR": "Username already exists"}

    @staticmethod
    def get_user_by_id(user_id: int):
        with get_session() as db_session:
            user = db_session.query(User).filter_by(User.id == user_id).first()
        return user

    @staticmethod
    def get_user_by_username(username: str):
        with get_session() as db_session:
            user = db_session.query(User).filter_by(User.username == username).first()
        return user

    @staticmethod
    def get_all_users():
        with get_session() as db_session:
            users = db_session.query(User).all()
        return users

    @staticmethod
    def delete_user_by_username(username: str):
        user = UserService.get_user_by_username(username)

        if user:
            with get_session() as db_session:
                db_session.delete(user)
                db_session.commit()
            return {"SUCCESS": f"User {username} was deleted"}
        else:
            print(f'User {username} not found!')
