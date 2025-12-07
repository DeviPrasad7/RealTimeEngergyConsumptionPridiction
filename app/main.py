from dotenv import load_dotenv
load_dotenv()

import logging
import json
from typing import List, Optional
import uuid
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from datetime import date
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from src.config import (
    MODELS_DIR,
    META_PATH,
    TSD_HISTORY_PATH,
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_PREDICTIONS_TABLE,
    create_dirs,
)
from app.features import load_history, make_timestamp, build_features

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

model = None
history = None
meta = {}
supabase_client: Optional[Client] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, history, meta, supabase_client
    logger.info("Starting up...")
    create_dirs()
    versions = sorted(MODELS_DIR.glob("model_v*.pkl"))
    if not versions:
        raise RuntimeError("No model versions available.")
    latest = versions[-1]
    model = joblib.load(latest)
    logger.info(f"Loaded model {latest}")
    if not TSD_HISTORY_PATH.exists():
        raise RuntimeError(f"History file not found at {TSD_HISTORY_PATH}. Run training first.")
    history = load_history(TSD_HISTORY_PATH)
    logger.info(f"Loaded history from {TSD_HISTORY_PATH}")
    
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    else:
        meta = {"version": None, "last_trained": None}

    if create_client and SUPABASE_URL and SUPABASE_KEY and SUPABASE_PREDICTIONS_TABLE:
        supabase_client = create_client(str(SUPABASE_URL), SUPABASE_KEY)
        logger.info("Supabase client initialized.")
    else:
        supabase_client = None
        logger.warning("Supabase client not initialized. Check environment variables.")
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

class DemandRequest(BaseModel):
    settlement_date: date
    settlement_period: int = Field(ge=1, le=48)

class DemandResponse(BaseModel):
    settlement_date: date
    settlement_period: int
    prediction: float

class HealthResponse(BaseModel):
    model_loaded: bool
    history_loaded: bool
    last_trained: str | None
    version: int | None
    status: str
    supabase_connected: bool

def log_to_supabase(request_id: str, items: List[DemandResponse]):
    if supabase_client is None:
        return
    rows = []
    for item in items:
        rows.append(
            {
                "request_id": request_id,
                "settlement_date": item.settlement_date.isoformat(),
                "settlement_period": item.settlement_period,
                "prediction": item.prediction,
            }
        )
    try:
        supabase_client.table(SUPABASE_PREDICTIONS_TABLE).upsert(
            rows, on_conflict="settlement_date,settlement_period"
        ).execute()
    except Exception as e:
        logger.error(f"Failed to log to Supabase: {e}")
        return

@app.post("/predict", response_model=DemandResponse)
def predict(req: DemandRequest, background_tasks: BackgroundTasks):
    try:
        ts = make_timestamp(req.settlement_date.isoformat(), req.settlement_period)
        X = build_features(ts, history)
        pred = float(model.predict(X)[0])
        resp = DemandResponse(
            settlement_date=req.settlement_date,
            settlement_period=req.settlement_period,
            prediction=pred
        )
        request_id = str(uuid.uuid4())
        background_tasks.add_task(log_to_supabase, request_id, [resp])
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error.")

@app.post("/predict_bulk", response_model=List[DemandResponse])
def predict_bulk(reqs: List[DemandRequest], background_tasks: BackgroundTasks):
    if not reqs:
        raise HTTPException(status_code=400, detail="Empty request list.")
    try:
        timestamps = [make_timestamp(r.settlement_date.isoformat(), r.settlement_period) for r in reqs]
        ts_index = pd.DatetimeIndex(timestamps)
        
        X = build_features(ts_index, history)
        preds = model.predict(X)
        
        responses = []
        for r, p in zip(reqs, preds):
            responses.append(
                DemandResponse(
                    settlement_date=r.settlement_date,
                    settlement_period=r.settlement_period,
                    prediction=float(p)
                )
            )
        
        request_id = str(uuid.uuid4())
        background_tasks.add_task(log_to_supabase, request_id, responses)
        return responses
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal bulk prediction error.")

@app.get("/health", response_model=HealthResponse)
def health():
    m_ok = model is not None
    h_ok = history is not None
    v = meta.get("version")
    t = meta.get("last_trained")
    s_ok = supabase_client is not None
    status = "ok" if (m_ok and h_ok) else "degraded"
    return HealthResponse(
        model_loaded=m_ok,
        history_loaded=h_ok,
        last_trained=t,
        version=v,
        status=status,
        supabase_connected=s_ok
    )