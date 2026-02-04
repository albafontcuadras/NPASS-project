# Configuration of the Dash UI and callbacks

from django_plotly_dash import DjangoDash
from dash import html, dcc, Input, Output, State
import dash
import dash_bootstrap_components as dbc
from .ml import predict


app = DjangoDash("npass_app", external_stylesheets=[dbc.themes.BOOTSTRAP])

# Helpers
def numeric_input(id_, value, min_=0, max_=100, step=0.1):
    return dcc.Input(
        id=id_,
        type="number",
        value=value,
        min=min_,
        max=max_,
        step=step,
        debounce=True,
        style={"width": "100%"},
    )

def slider(id_, min_, max_, step, value):
    return dcc.Slider(
        id=id_, min=min_, max=max_, step=step, value=value,
        tooltip={"always_visible": True}
    )

def ratio_slider(id_, value=0.2):
    return dcc.Slider(
        id=id_, min=0, max=100, step=1, value=int(value * 100),
        marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
        tooltip={"always_visible": True}
    )

# Layout 
app.layout = dbc.Container([
    html.H2("Nurse Perceived Adequacy of Staffing Scale (NPASS score prediction)"),
    html.P(
        "Enter the details for the upcoming shift or day to see the predicted NPASS score. "
        "Then click Predict to see the results."
    ),

    dbc.Row([
        # Temporal
        dbc.Col([
            html.H4("Temporal"),
            html.P("Moving averages and recent NPASS scores (t-1, t-2)."),
            dbc.Label("NPASS moving avg (3)"),
            numeric_input("npass_moving_avg_3", 5.0, 0, 10, 0.1),
            dbc.Label("NPASS t-1"),
            numeric_input("npass_t_1", 5.0, 0, 10, 0.1),
            dbc.Label("NPASS t-2"),
            numeric_input("npass_t_2", 5.0, 0, 10, 0.1),
        ], md=4),

        # Operational
        dbc.Col([
            html.H4("Operational"),
            html.P("Ratios shown as percentages; they will be converted to 0–1 for the model."),
            dbc.Label("Avg patient count"),
            numeric_input("avg_patient_count", 0, 0, 80, 1),
            dbc.Label("Admission ratio"),
            ratio_slider("admission_ratio", 0.50),
            dbc.Label("Discharge ratio"),
            ratio_slider("discharge_ratio", 0.50),
            dbc.Label("Transfer ratio"),
            ratio_slider("transfer_ratio", 0.50),
            dbc.Label("ICU ratio"),
            ratio_slider("icu_ratio", 0.50),
            dbc.Label("Emergency admission ratio"),
            ratio_slider("emergency_admission_ratio", 0.50),
            dbc.Label("Pct elderly 70+"),
            ratio_slider("pct_elderly_70plus", 0.50),
            dbc.Label("Long-stay ratio 7+"),
            ratio_slider("long_stay_ratio_7plus", 0.50),
        ], md=5),

        # Structural
        dbc.Col([
            html.H4("Structural"),
            html.P("Organisation and scheduling context."),
            dbc.Label("Hospital ID"),
            dcc.Dropdown(
                id="hospital_id",
                options=[{"label": h, "value": h} for h in ["H1", "H2", "H3", "H4", "H5"]],
                value="H1", clearable=False
            ),
            dbc.Label("Department type"),
            dcc.Dropdown(
                id="department_type",
                options=[{"label": d, "value": d} for d in
                         ["Emergency", "Surgery", "Cardiology", "Oncology", "Pediatrics", "General Medicine"]],
                value="General Medicine", clearable=False
            ),
            dbc.Label("Day of week"),
            dcc.Dropdown(
                id="day_of_week",
                options=[{"label": d, "value": d} for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]],
                value="Mon", clearable=False
            ),
            dbc.Label("Shift type"),
            dcc.Dropdown(
                id="shift_type",
                options=[{"label": s, "value": s} for s in ["Day", "Evening", "Night"]],
                value="Day", clearable=False
            ),
        ], md=3),
    ], className="gy-3"),

    html.Hr(),
    dbc.Button("Predict", id="predict_btn", color="primary"),
    html.Div(id="prediction_output", style={"marginTop": "1rem", "fontWeight": "bold"})
], fluid=True)

# Callback
@app.callback(
    Output("prediction_output", "children"),
    Input("predict_btn", "n_clicks"),
    State("npass_moving_avg_3", "value"),
    State("npass_t_1", "value"),
    State("npass_t_2", "value"),
    State("day_of_week", "value"),
    State("avg_patient_count", "value"),
    State("admission_ratio", "value"),
    State("discharge_ratio", "value"),
    State("transfer_ratio", "value"),
    State("icu_ratio", "value"),
    State("emergency_admission_ratio", "value"),
    State("pct_elderly_70plus", "value"),
    State("long_stay_ratio_7plus", "value"),
    State("hospital_id", "value"),
    State("department_type", "value"),
    State("shift_type", "value"),
    prevent_initial_call=True
)
def on_predict(n_clicks, mavg, t1, t2, dow, cnt, adm, dis, trf, icu, emer, elderly, longstay, hosp, dept, shift):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

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

    pred = predict(payload)
    try:
        return f"NPASS score prediction for the next shift is: {float(pred):.3f}"
    except Exception:
        return f"Prediction: {pred}"
