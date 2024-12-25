import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)


RISK_MASTER_ROOT = os.getenv("RISK_MASTER_ROOT")
STYLES_PATH = f"{RISK_MASTER_ROOT}{os.getenv("STYLES_PATH")}"

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
