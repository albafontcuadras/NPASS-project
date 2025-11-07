from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from typing import List, Dict, Any
from pathlib import Path
import logging
import json

import joblib
import numpy as np
import pandas as pd

# FastAPI
app = FastAPI(title="Spotify Model API", version="1.0.0")

# Artifacts
ART_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ART_DIR / "model.joblib"
META_PATH = ART_DIR / "metadata.json"

model = None
meta: Dict[str, Any] = {}
FEATURES: List[str] = []

class PredictRequest(BaseModel):
    track_popularity: conint(ge=0, le=100)
    key: conint(ge=0, le=11)
    mode: conint(ge=0, le=1)
    duration_ms: conint(ge=0)
    danceability: confloat(ge=0, le=1)
    energy: confloat(ge=0, le=1)
    speechiness: confloat(ge=0, le=1)
    acousticness: confloat(ge=0, le=1)
    instrumentalness: confloat(ge=0, le=1)
    liveness: confloat(ge=0, le=1)
    valence: confloat(ge=0, le=1)
    loudness: float
    tempo: float

class PredictResponse(BaseModel):
    prediction: int

@app.on_event("startup")
def load():
    global model, meta, FEATURES
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
        if not META_PATH.exists():
            raise FileNotFoundError(f"Missing metadata at {META_PATH}")

        model = joblib.load(MODEL_PATH)

        with open(META_PATH) as f:
            meta = json.load(f)

        FEATURES = meta.get("features") or []
        if not FEATURES:
            raise RuntimeError("No 'features' list found in metadata.json")

        req_fields = set(PredictRequest.model_fields.keys())
        missing = [f for f in FEATURES if f not in req_fields]
        if missing:
            raise RuntimeError(
                f"API schema missing fields used by the model: {missing}"
            )

        logging.getLogger("uvicorn").info(
            f"Loaded model '{meta.get('model_name')}' with {len(FEATURES)} features."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")

# Health check
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {
        "ready": model is not None,
        "model_name": meta.get("model_name"),
        "created_at": meta.get("created_at"),
        "sklearn_version_artifact": meta.get("sklearn_version"),
    }

# Predicting
@app.post("/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        row = {f: getattr(payload, f) for f in FEATURES}
        X = pd.DataFrame([row], columns=FEATURES)
        y_pred = int(model.predict(X)[0])
        return PredictResponse(prediction=y_pred)
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


from starlette.middleware.wsgi import WSGIMiddleware
import dash
from dash import html, dcc, Input, Output, State
import requests
import os

def make_dash_app():
    dash_app = dash.Dash(
        __name__,
        requests_pathname_prefix="/ui/",
        suppress_callback_exceptions=True,
    )

    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

    fields = [
        ("track_popularity", 50),
        ("danceability", 0.5),
        ("energy", 0.5),
        ("key", 5),
        ("loudness", -10.0),
        ("mode", 1),
        ("speechiness", 0.05),
        ("acousticness", 0.1),
        ("instrumentalness", 0.0),
        ("liveness", 0.1),
        ("valence", 0.5),
        ("tempo", 120.0),
        ("duration_ms", 180000),
    ]

    INT_FIELDS = {"track_popularity", "key", "mode", "duration_ms"}
    UNIT_FIELDS = {"danceability", "energy", "speechiness", "acousticness",
                   "instrumentalness", "liveness", "valence"}

    def input_block(name, default):
        return html.Div([
            html.Label(name),
            dcc.Input(id=name, type="number", value=default, debounce=True, style={"width": "100%"})
        ], style={"marginBottom": "10px"})

    dash_app.layout = html.Div([
        html.H2("🎧 Spotify Song Classifier"),
        html.Div([input_block(n, d) for n, d in fields]),
        html.Button("Predict", id="predict-btn", style={"marginTop": "15px"}),
        html.Div(id="result", style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "18px"}),
        html.Div("API docs at /docs", style={"marginTop":"8px", "opacity":0.7})
    ], style={"maxWidth": "520px", "margin": "40px auto", "fontFamily": "Arial, sans-serif"})

    @dash_app.callback(
        Output("result", "children"),
        Input("predict-btn", "n_clicks"),
        [State(n, "value") for n, _ in fields]
    )
    def do_predict(n_clicks, *values):
        if not n_clicks:
            return ""
        payload = {}
        for (name, _), val in zip(fields, values):
            if val is None:
                return f"'{name}' is empty. Please enter a value."
            if name in INT_FIELDS:
                payload[name] = int(round(float(val)))
            else:
                x = float(val)
                if name in UNIT_FIELDS:
                    x = max(0.0, min(1.0, x))
                payload[name] = x
        try:
            r = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=10)
            r.raise_for_status()
            pred = r.json().get("prediction")
            return f"Predicted label: {pred}"
        except requests.HTTPError:
            return f"{r.status_code} — {r.text}"
        except Exception as e:
            return f"Error: {e}"

    return dash_app

_dash = make_dash_app()
app.mount("/ui", WSGIMiddleware(_dash.server))

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
