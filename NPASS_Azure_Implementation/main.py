# FastAPI and Dash configuration

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import random
import time

# FastAPI
app = FastAPI(title="NPASS Implementation", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Schema
class NPASSInput(BaseModel):
    # Temporal
    npass_moving_avg_3: float = Field(5.0, ge=0, le=10)
    npass_t_1: float = Field(5.0, ge=0, le=10)
    npass_t_2: float = Field(5.0, ge=0, le=10)
    day_of_week: Literal["Mon","Tue","Wed","Thu","Fri","Sat","Sun"] = "Mon"
    # Operational
    avg_patient_count: int = Field(20, ge=0, le=80)
    admission_ratio: float = Field(0.2, ge=0.0, le=1.0)
    discharge_ratio: float = Field(0.2, ge=0.0, le=1.0)
    transfer_ratio: float = Field(0.1, ge=0.0, le=1.0)
    icu_ratio: float = Field(0.15, ge=0.0, le=1.0)
    emergency_admission_ratio: float = Field(0.2, ge=0.0, le=1.0)
    pct_elderly_70plus: float = Field(30.0, ge=0.0, le=100.0)
    long_stay_ratio_7plus: float = Field(0.25, ge=0.0, le=1.0)
    # Structural
    hospital_id: Literal["H1","H2","H3","H4","H5"] = "H1"
    department_type: Literal["Emergency","Surgery","Cardiology","Oncology","Pediatrics","General Medicine"] = "General Medicine"
    shift_type: Literal["Day","Evening","Night"] = "Day"

@app.post("/v1/predict")
def predict(payload: NPASSInput):
    return {"prediction": (random.random() * 3.0) - 1.0}


# Dash
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.BOOTSTRAP]


dash_app = dash.Dash(
    __name__,
    requests_pathname_prefix="/ui/",
    external_stylesheets=external_stylesheets,
    serve_locally=True, 
)

def numeric_input(id_, value, min_=0, max_=100, step=0.1):
    return dcc.Input(
        id=id_,
        type="number",
        value=value,
        min=min_,
        max=max_,
        step=step,
        debounce=True,
        style={"width": "100%"}
    )


def slider(id_, min_, max_, step, value):
    return dcc.Slider(id=id_, min=min_, max=max_, step=step, value=value, tooltip={"always_visible": True})

def ratio_slider(id_, value=0.2):
    return dcc.Slider(
        id=id_, min=0, max=100, step=1, value=int(value*100),
        marks={0:"0%",25:"25%",50:"50%",75:"75%",100:"100%"},
        tooltip={"always_visible": True}
    )

dash_app.layout = dbc.Container([
    html.H2("Nurse Perceived Adequacy of Staffing Scale (NPASS score prediction)"),
    html.P("Enter the details for the upcoming shift or day to see the estimated NPASS score. Then just click Predict to see the results."),

    dbc.Row([
        dbc.Col([
            html.H4("Temporal"),
            html.P("This section shows the moving average of NPASS scores from past days or shifts, these are provided automatically by the algorithm."),
            dbc.Label("NPASS moving avg (3)"),
            numeric_input("npass_moving_avg_3", 5.0, 0, 10, 0.1),
            dbc.Label("NPASS t-1"),
            numeric_input("npass_t_1", 5.0, 0, 10, 0.1),
            dbc.Label("NPASS t-2"),
            numeric_input("npass_t_2", 5.0, 0, 10, 0.1),
        ], md=4),

        dbc.Col([
            html.H4("Operational"),
            html.P("These fields capture key operational metrics that reflect the overall patient flow and workload in the unit."),
            dbc.Label("Average patient count"),
            numeric_input("avg_patient_count", 20, 0, 80, 1),
            dbc.Label("Admission ratio"),
            ratio_slider("admission_ratio", 0.2),
            dbc.Label("Discharge ratio"),
            ratio_slider("discharge_ratio", 0.2),
            dbc.Label("Transfer ratio"),
            ratio_slider("transfer_ratio", 0.1),
            dbc.Label("ICU ratio"),
            ratio_slider("icu_ratio", 0.15),
            dbc.Label("Emergency admission ratio"),
            ratio_slider("emergency_admission_ratio", 0.2),
            dbc.Label("% elderly patients (70+)"),
            dcc.Slider(id="pct_elderly_70plus", min=0, max=100, step=1, value=30,
                       marks={0:"0%",25:"25%",50:"50%",75:"75%",100:"100%"},
                       tooltip={"always_visible": True}),
            dbc.Label("Long-stay ratio (7+ days)"),
            ratio_slider("long_stay_ratio_7plus", 0.25),
        ], md=5),

        dbc.Col([
            html.H4("Structural"),
            html.P("Basic organisational and scheduling details that are used to contextualise the staffing patterns and the NPASS prediction."),
            dbc.Label("Hospital ID"),
            dcc.Dropdown(
                id="hospital_id",
                options=[{"label":h,"value":h} for h in ["H1","H2","H3","H4","H5"]],
                value="H1", clearable=False
            ),
            dbc.Label("Department type"),
            dcc.Dropdown(
                id="department_type",
                options=[{"label":d,"value":d} for d in
                         ["Emergency","Surgery","Cardiology","Oncology","Pediatrics","General Medicine"]],
                value="General Medicine", clearable=False
            ),
            dbc.Label("Day of week"),
            dcc.Dropdown(
                id="day_of_week",
                options=[{"label":d,"value":d} for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]],
                value="Mon", clearable=False
            ),
            dbc.Label("Shift type"),
            dcc.Dropdown(
                id="shift_type",
                options=[{"label":s,"value":s} for s in ["Day","Evening","Night"]],
                value="Day", clearable=False
            ),
        ], md=3),
    ], className="mt-3"),

    html.Hr(),
    dbc.Button("Predict", id="predict_btn", color="primary"),
    html.Div(id="prediction_output", className="mt-3", style={"fontSize":"1.25rem"}),
], fluid=True)

import os
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


@dash_app.callback(
    Output("prediction_output","children"),
    Input("predict_btn","n_clicks"),
    State("npass_moving_avg_3","value"),
    State("npass_t_1","value"),
    State("npass_t_2","value"),
    State("day_of_week","value"),
    State("avg_patient_count","value"),
    State("admission_ratio","value"),
    State("discharge_ratio","value"),
    State("transfer_ratio","value"),
    State("icu_ratio","value"),
    State("emergency_admission_ratio","value"),
    State("pct_elderly_70plus","value"),
    State("long_stay_ratio_7plus","value"),
    State("hospital_id","value"),
    State("department_type","value"),
    State("shift_type","value"),
    prevent_initial_call=True
)
def on_predict(n_clicks, mavg, t1, t2, dow, cnt, adm, dis, trf, icu, emer, elderly, longstay, hosp, dept, shift):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    import requests
    payload = {
        "npass_moving_avg_3": float(mavg),
        "npass_t_1": float(t1),
        "npass_t_2": float(t2),
        "day_of_week": dow,
        "avg_patient_count": int(cnt),
        "admission_ratio": float(adm) / 100.0,
        "discharge_ratio": float(dis) / 100.0,
        "transfer_ratio": float(trf) / 100.0,
        "icu_ratio": float(icu) / 100.0,
        "emergency_admission_ratio": float(emer) / 100.0,
        "pct_elderly_70plus": float(elderly),
        "long_stay_ratio_7plus": float(longstay) / 100.0,
        "hospital_id": hosp,
        "department_type": dept,
        "shift_type": shift,
    }
    r = requests.post(f"{API_BASE}/v1/predict", json=payload, timeout=5)
    r.raise_for_status()
    return f"NPASS score prediction for the next shift is: {r.json()['prediction']:.3f}"

app.mount("/ui", WSGIMiddleware(dash_app.server))

@app.get("/")
def root():
    return {"message": "See /docs for the API and /ui/ for the demo UI"}

@app.on_event("startup")
async def _show_routes_and_dash():
    print("Mounted routes:", [getattr(r, "path", str(r)) for r in app.router.routes])
    print("Dash prefixes => requests_pathname_prefix:",
          dash_app.config.requests_pathname_prefix,
          " routes_pathname_prefix:", dash_app.config.routes_pathname_prefix)
    print("Startup marker:", time.time())
