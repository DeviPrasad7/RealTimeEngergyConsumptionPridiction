import os
from pathlib import Path

BASE = Path(os.getenv("BASE_PATH", ".")).resolve()

RAW_DATA_PATH = BASE / os.getenv("RAW_DATA_PATH", "data/raw/historic_demand_2009_2024.csv")

MODELS_DIR = BASE / os.getenv("MODELS_DIR", "models")
ARTIFACTS_DIR = BASE / os.getenv("ARTIFACTS_DIR", "artifacts")
SNAPSHOTS_DIR = ARTIFACTS_DIR / "snapshots"
META_PATH = ARTIFACTS_DIR / "train_state.json"
TSD_HISTORY_PATH = ARTIFACTS_DIR / "tsd_history.parquet"

def create_dirs():
    """Creates the directories required for the project."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = 0.15
MIN_NEW_ROWS = 100

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_ANON_KEY"))
SUPABASE_RAW_TABLE = os.getenv("SUPABASE_RAW_TABLE", "demand_raw")
SUPABASE_PREDICTIONS_TABLE = os.getenv("SUPABASE_PREDICTIONS_TABLE", "demand_predictions")
SUPABASE_PROCESSED_TABLE = os.getenv("SUPABASE_PROCESSED_TABLE", "demand_processed")

RAW_DATA_PAGE_SIZE = int(os.getenv("RAW_DATA_PAGE_SIZE", 1000))
PROCESSED_DATA_CHUNK_SIZE = int(os.getenv("PROCESSED_DATA_CHUNK_SIZE", 500))