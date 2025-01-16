import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)


RISK_MASTER_ROOT = os.getenv("RISK_MASTER_ROOT")
STYLES_PATH = f"{RISK_MASTER_ROOT}{os.getenv("STYLES_PATH")}"
DATA_SETS_PATH = f"{RISK_MASTER_ROOT}{os.getenv("DATA_SETS_PATH")}"
PRICE_PREDICTION_SAVED_MODELS_PATH = f"{RISK_MASTER_ROOT}{os.getenv("PRICE_PREDICTION_SAVED_MODELS_PATH")}"

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

GOLD_DATA_SET = os.getenv("GOLD_DATA_SET")
BITCOIN_DATA_SET = os.getenv("BITCOIN_DATA_SET")
SILVER_DATA_SET = os.getenv("SILVER_DATA_SET")
ETH_DATA_SET = os.getenv("ETH_DATA_SET")
DOGE_DATA_SET = os.getenv("DOGE_DATA_SET")