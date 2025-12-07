import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_PATH = Path(os.getenv("BASE_PATH", ".")).resolve()
RAW_DATA_PATH = BASE_PATH / os.getenv("RAW_DATA_PATH", "data/raw/historic_demand_2009_2024.csv")
PROCESSED_DIR = BASE_PATH / os.getenv("PROCESSED_DIR", "data/processed")
MODEL_DIR = BASE_PATH / os.getenv("MODEL_DIR", "model")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {"settlement_date", "settlement_period", "tsd"}
DROP_COLUMNS = ["scottish_transfer", "viking_flow", "greenlink_flow", "nsl_flow", "eleclink_flow"]

TRAIN_STATE_PATH = MODEL_DIR / "train_state.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_PATH = MODEL_DIR / "final_xgb_model.pkl"