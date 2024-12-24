import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from env_loader import *

db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def initialize_database(base):
    engine = create_engine(db_url)
    base.metadata.create_all(engine)


def db_connection():
    try:
        connection = psycopg2.connect(
            host = DB_HOST,
            port = DB_PORT,
            database = DB_NAME,
            user = DB_USER,
            password = DB_PASSWORD
        )
        return connection
    except psycopg2.Error as e:
        print(f'DB connection ERROR: {e}')
        raise


def get_session():
    try:
        return sessionmaker(bind=get_engine())()
    except Exception as e:
        print(f'DB connection ERROR: {e}')
        raise


def get_engine():
    try:
        return create_engine(db_url)
    except Exception as e:
        print(f"DB connection ERROR: {e}")
        raise
