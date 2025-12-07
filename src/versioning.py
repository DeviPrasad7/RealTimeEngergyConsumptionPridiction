import json
import hashlib
from datetime import datetime
import pandas as pd
from pathlib import Path
from src.config import MODELS_DIR, META_PATH, SNAPSHOTS_DIR

def _hash_df(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode()
    return hashlib.sha256(b).hexdigest()

def load_metadata():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"version": 0, "last_hash": None, "last_trained": None}

def save_metadata(meta: dict):
    META_PATH.write_text(json.dumps(meta, indent=2))

def create_snapshot(df: pd.DataFrame) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = SNAPSHOTS_DIR / f"snapshot_{ts}.parquet"
    df.to_parquet(p)
    return p

def next_model_path() -> Path:
    meta = load_metadata()
    v = meta["version"] + 1
    return MODELS_DIR / f"model_v{v}.pkl"

def update_version(hash_val: str):
    meta = load_metadata()
    meta["version"] += 1
    meta["last_hash"] = hash_val
    meta["last_trained"] = datetime.now().isoformat()
    save_metadata(meta)

def is_new_data(df: pd.DataFrame, min_rows: int) -> bool:
    meta = load_metadata()
    new_hash = _hash_df(df)
    if meta["last_hash"] is None:
        return True
    if new_hash != meta["last_hash"] and len(df) >= min_rows:
        return True
    return False