import json
from typing import List, Optional
import uuid
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from src.config import (
    MODELS_DIR,
    META_PATH,
    TSD_HISTORY_PATH,
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_PREDICTIONS_TABLE,
)
from app.features import load_history, make_timestamp, build_features

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

app = FastAPI()

model = None
history = None
meta = {}
supabase_client: Optional["Client"] = None

class DemandRequest(BaseModel):
    settlement_date: str
    settlement_period: int = Field(ge=1, le=48)

class DemandResponse(BaseModel):
    settlement_date: str
    settlement_period: int
    prediction: float

class HealthResponse(BaseModel):
    model_loaded: bool
    history_loaded: bool
    last_trained: str | None
    version: int | None
    status: str
    supabase_connected: bool

@app.on_event("startup")
def startup():
    global model, history, meta, supabase_client
    versions = sorted(MODELS_DIR.glob("model_v*.pkl"))
    if not versions:
        raise RuntimeError("No model versions available.")
    latest = versions[-1]
    model = joblib.load(latest)
    if not TSD_HISTORY_PATH.exists():
        raise RuntimeError(f"History file not found at {TSD_HISTORY_PATH}. Run training first.")
    history = load_history(TSD_HISTORY_PATH)
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    else:
        meta = {"version": None, "last_trained": None}
    if create_client is not None and SUPABASE_URL and SUPABASE_KEY and SUPABASE_PREDICTIONS_TABLE:
        supabase_client = create_client(str(SUPABASE_URL), SUPABASE_KEY)
    else:
        supabase_client = None

def log_to_supabase(request_id: str, items: List[DemandResponse]):
    if supabase_client is None:
        return
    rows = []
    for item in items:
        rows.append(
            {
                "request_id": request_id,
                "settlement_date": item.settlement_date,
                "settlement_period": item.settlement_period,
                "prediction": item.prediction,
            }
        )
    try:
        supabase_client.table(SUPABASE_PREDICTIONS_TABLE).insert(rows).execute()
    except Exception:
        return

@app.post("/predict", response_model=DemandResponse)
def predict(req: DemandRequest, background_tasks: BackgroundTasks):
    global model, history
    try:
        ts = make_timestamp(req.settlement_date, req.settlement_period)
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
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal prediction error.")

@app.post("/predict_bulk", response_model=List[DemandResponse])
def predict_bulk(reqs: List[DemandRequest], background_tasks: BackgroundTasks):
    global model, history
    if not reqs:
        raise HTTPException(status_code=400, detail="Empty request list.")
    try:
        frames = []
        meta_rows = []
        for r in reqs:
            ts = make_timestamp(r.settlement_date, r.settlement_period)
            X_i = build_features(ts, history)
            frames.append(X_i)
            meta_rows.append((r.settlement_date, r.settlement_period))
        X = frames[0] if len(frames) == 1 else pd.concat(frames)
        preds = model.predict(X)
        responses = []
        for (settlement_date, settlement_period), p in zip(meta_rows, preds):
            responses.append(
                DemandResponse(
                    settlement_date=settlement_date,
                    settlement_period=settlement_period,
                    prediction=float(p)
                )
            )
        request_id = str(uuid.uuid4())
        background_tasks.add_task(log_to_supabase, request_id, responses)
        return responses
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
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