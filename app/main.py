import logging
import json
from typing import List, Optional
import uuid
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from datetime import date, datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
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

def log_request_metrics(
    timestamp: datetime,
    request_id: str,
    path: str,
    status: str,
    latency_ms: float,
    model_version: Optional[str],
    n_predictions: Optional[int] = None,
):
    log_details = [
        f"  TIMESTAMP: {timestamp.isoformat()}",
        f"  REQUEST_ID: {request_id}",
        f"  PATH: {path}",
        f"  STATUS: {status}",
        f"  LATENCY_MS: {latency_ms:.2f}",
        f"  MODEL_VERSION: {model_version or 'N/A'}",
    ]
    if n_predictions is not None:
        log_details.append(f"  N_PREDICTIONS: {n_predictions}")
    
    logger.info("Request Metrics:\n" + "\n".join(log_details))

model = None
history = None
meta = {}
supabase_client: Optional[Client] = None # type: ignore

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
def predict(req: DemandRequest, background_tasks: BackgroundTasks, request: Request):
    start_time = datetime.now()
    request_id = str(uuid.uuid4())
    status = "failed"
    n_predictions = 1

    try:
        ts = make_timestamp(req.settlement_date.isoformat(), req.settlement_period)
        X = build_features(ts, history)
        pred = float(model.predict(X)[0])
        resp = DemandResponse(
            settlement_date=req.settlement_date,
            settlement_period=req.settlement_period,
            prediction=pred
        )
        # Use the same request_id for Supabase logging
        background_tasks.add_task(log_to_supabase, request_id, [resp])
        status = "success"
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error.")
    finally:
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        model_version = meta.get("version")
        log_request_metrics(
            timestamp=start_time,
            request_id=request_id,
            path=request.url.path,
            status=status,
            latency_ms=latency_ms,
            model_version=model_version,
            n_predictions=n_predictions,
        )

@app.post("/predict_bulk", response_model=List[DemandResponse])
def predict_bulk(reqs: List[DemandRequest], background_tasks: BackgroundTasks, request: Request):
    start_time = datetime.now()
    request_id = str(uuid.uuid4())
    status = "failed"
    n_predictions = len(reqs)

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
        
        # Use the same request_id for Supabase logging
        background_tasks.add_task(log_to_supabase, request_id, responses)
        status = "success"
        return responses
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal bulk prediction error.")
    finally:
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        model_version = meta.get("version")
        log_request_metrics(
            timestamp=start_time,
            request_id=request_id,
            path=request.url.path,
            status=status,
            latency_ms=latency_ms,
            model_version=model_version,
            n_predictions=n_predictions,
        )

@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    start_time = datetime.now()
    request_id = str(uuid.uuid4())
    status = "failed"
    
    try:
        m_ok = model is not None
        h_ok = history is not None
        v = meta.get("version")
        t = meta.get("last_trained")
        s_ok = supabase_client is not None
        endpoint_status = "ok" if (m_ok and h_ok) else "degraded"
        
        resp = HealthResponse(
            model_loaded=m_ok,
            history_loaded=h_ok,
            last_trained=t,
            version=v,
            status=endpoint_status,
            supabase_connected=s_ok
        )
        status = "success"
        return resp
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal health check error.")
    finally:
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        # For health endpoint, model_version and n_predictions are not directly applicable in the same way as predict endpoints
        # We can pass None for these or specific values if desired.
        log_request_metrics(
            timestamp=start_time,
            request_id=request_id,
            path=request.url.path,
            status=status,
            latency_ms=latency_ms,
            model_version=meta.get("version"), # Still useful to know which model version is reported in health
            n_predictions=None, 
        )