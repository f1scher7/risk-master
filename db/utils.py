import os
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=env_path)

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

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
        engine = get_engine()
        return sessionmaker(bind=engine)()
    except Exception as e:
        print(f'DB connection ERROR: {e}')
        raise

def get_engine():
    try:
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        print(f"DB connection ERROR: {e}")
        raise
