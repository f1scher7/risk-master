from db.models import Base
from db.db_connection import initialize_database
from services.user_service import UserService

if __name__ == "__main__":
    # initialize_database(Base)

    user = UserService.create_user("kaziof1scher7", "test")
