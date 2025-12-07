from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import joblib
import logging
from src.config import (
    TSD_HISTORY_PATH,
    DRIFT_THRESHOLD,
    MIN_NEW_ROWS,
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_RAW_TABLE,
    SUPABASE_PROCESSED_TABLE,
    RAW_DATA_PAGE_SIZE,
    PROCESSED_DATA_CHUNK_SIZE,
    create_dirs,
)
from src.data_pipeline import preprocess, clean_raw_df
from src.modeling import make_splits, train_best, FEATURES, TARGET
from src.versioning import (
    is_new_data,
    next_model_path,
    create_snapshot,
    update_version,
    load_metadata,
    _hash_df,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

def drift_score(df_old: pd.DataFrame, df_new: pd.DataFrame) -> float:
    a = df_old["tsd"].tail(2000).values
    b = df_new["tsd"].tail(2000).values
    if len(a) == 0 or len(b) == 0:
        return 1.0
    return float(abs(a.mean() - b.mean()) / max(a.mean(), 1e-6))

def build_history(df: pd.DataFrame, end_date: pd.Timestamp):
    start = end_date - pd.Timedelta("1095 days")
    h = df[df.index >= start][["tsd"]]
    h.to_parquet(TSD_HISTORY_PATH)

def load_raw_supabase() -> pd.DataFrame:
    if create_client is None or not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_RAW_TABLE:
        raise RuntimeError("Supabase is not configured for training.")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    logger.info("Downloading raw data from Supabase...")
    all_rows = []
    start_index = 0
    
    while True:
        try:
            logger.info(f"Downloading rows from {start_index} to {start_index + RAW_DATA_PAGE_SIZE - 1}...")
            res = client.table(SUPABASE_RAW_TABLE).select("*").range(start_index, start_index + RAW_DATA_PAGE_SIZE - 1).execute()
            rows = res.data
            if not rows:
                break
            all_rows.extend(rows)
            start_index += len(rows)
        except Exception as e:
            logger.error(f"Failed to download raw data from Supabase: {e}", exc_info=True)
            raise
    
    if not all_rows:
        raise RuntimeError("Supabase raw table is empty.")
        
    logger.info(f"Total rows downloaded: {len(all_rows)}")
    df = pd.DataFrame(all_rows)
    return clean_raw_df(df)

def upload_processed_supabase(df: pd.DataFrame, client: Client):
    if df.empty:
        logger.info("Processed dataframe is empty. Nothing to upload.")
        return

    df_upload = df.copy()
    cols_to_upload = FEATURES + [TARGET, "settlement_date"]
    cols_to_upload = [col for col in cols_to_upload if col in df_upload.columns]
    df_upload = df_upload[cols_to_upload]

    df_upload.columns = df_upload.columns.str.lower()
    df_upload["timestamp"] = pd.to_datetime(df_upload.index).to_series().dt.strftime('%Y-%m-%dT%H:%M:%S')
    df_upload["settlement_date"] = df_upload["settlement_date"].astype(str)

    rows = df_upload.reset_index(drop=True).to_dict(orient="records")
    
    for i in range(0, len(rows), PROCESSED_DATA_CHUNK_SIZE):
        chunk = rows[i:i + PROCESSED_DATA_CHUNK_SIZE]
        try:
            logger.info(f"Upserting rows from {i} to {i + len(chunk) - 1}...")
            client.table(SUPABASE_PROCESSED_TABLE).upsert(chunk, on_conflict="timestamp").execute()
            logger.info(f"Successfully upserted {len(chunk)} rows to {SUPABASE_PROCESSED_TABLE}")
        except Exception as e:
            logger.error(f"Failed to upsert a chunk of processed data to Supabase: {e}", exc_info=True)
            break

def get_processed_data(client: Client) -> pd.DataFrame | None:
    """Gets the processed data, using cached data if possible."""
    try:
        df_raw = load_raw_supabase()
        logger.info(f"Loaded {len(df_raw)} rows from Supabase.")
    except Exception as e:
        raise RuntimeError(f"Failed to load raw data: {e}")

    if not is_new_data(df_raw, MIN_NEW_ROWS):
        logger.info("No new data detected. Attempting to load processed data from Supabase.")
        try:
            res = client.table(SUPABASE_PROCESSED_TABLE).select("*").execute()
            df_processed = pd.DataFrame(res.data)
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
            df_processed = df_processed.set_index('timestamp').sort_index()
            logger.info(f"Loaded {len(df_processed)} rows of processed data from Supabase.")
            return df_processed
        except Exception as e:
            logger.warning(f"Could not load processed data: {e}. Reprocessing raw data.", exc_info=True)

    logger.info("New data detected, reprocessing all data.")
    df_processed = preprocess(df_raw)
    
    if client:
        upload_processed_supabase(df_processed.copy(), client)

    create_snapshot(df_raw)
    update_version(_hash_df(df_raw))
    
    return df_processed

def train():
    """Main training function."""
    create_dirs()
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase is not configured. Please set SUPABASE_URL and SUPABASE_KEY.")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    df = get_processed_data(client)

    if df is None or df.empty:
        logger.info("No data available for training. Exiting.")
        return None

    meta = load_metadata()
    if meta["last_hash"]:
        df_old = pd.read_parquet(TSD_HISTORY_PATH) if TSD_HISTORY_PATH.exists() else None
        if df_old is not None:
            # Pass the raw dataframe to drift_score
            df_raw = load_raw_supabase()
            d = drift_score(df_old, df_raw)
            if d < DRIFT_THRESHOLD:
                logger.info(f"Drift score {d:.4f} below threshold. Skipping training.")
                return None

    train_df, test_df, hold_df = make_splits(df)
    model, rmse, r2 = train_best(train_df, test_df, hold_df)
    mpath = next_model_path()
    joblib.dump(model, mpath)
    
    end = df.index.max()
    build_history(df, end)
    
    return {"model_path": str(mpath), "rmse": rmse, "r2": r2}

if __name__ == "__main__":
    result = train()
    if result is None:
        logger.info("No new model trained.")
    else:
        logger.info("Training complete.")
        logger.info(f"Model saved to: {result['model_path']}")
        logger.info(f"Test RMSE: {result['rmse']:.4f}")
        logger.info(f"Test R² Score: {result['r2']:.4f}")