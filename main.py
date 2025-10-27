
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat
from typing import Dict, Any
import joblib, numpy as np, json
from pathlib import Path

# FastAPI access
app = FastAPI(title="Spotify Model API", version="1.0.0")

ART_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ART_DIR / "model.joblib"
META_PATH = ART_DIR / "metadata.json"

model = None
meta: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    track_popularity: conint(ge=0, le=100)
    danceability: confloat(ge=0, le=1)
    energy: confloat(ge=0, le=1)
    key: conint(ge=0, le=11)
    loudness: float
    mode: conint(ge=0, le=1)
    speechiness: confloat(ge=0, le=1)
    acousticness: confloat(ge=0, le=1)
    instrumentalness: confloat(ge=0, le=1)
    liveness: confloat(ge=0, le=1)
    valence: confloat(ge=0, le=1)
    tempo: float
    duration_ms: conint(ge=0)

class PredictResponse(BaseModel):
    prediction: int

@app.on_event("startup")
def load():
    global model, meta
    try:
        model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"ready": model is not None, "model_name": meta.get("model_name"), "version": meta.get("created_at")}

@app.post("/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        feats = ["track_popularity", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
        x = np.array([[getattr(payload, f) for f in feats]], dtype=float)
        y_pred = int(model.predict(x)[0])
        return PredictResponse(prediction=y_pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
