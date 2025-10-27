import dash
from dash import html, dcc, Input, Output, State
import requests
import os

# API URL — FastAPI server
API_URL = os.getenv("API_URL", "http://localhost:8000")

app = dash.Dash(__name__)
server = app.server 

# Input fields with default values
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

def input_block(name, default):
    return html.Div([
        html.Label(name),
        dcc.Input(
            id=name,
            type="number",
            value=default,
            debounce=True,
            style={"width": "100%"}
        )
    ], style={"marginBottom": "10px"})

app.layout = html.Div([
    html.H2("🎧 Spotify Song Classifier"),
    html.Div([input_block(n, d) for n, d in fields]),
    html.Button("Predict", id="predict-btn", style={"marginTop": "15px"}),
    html.Div(id="result", style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "18px"})
], style={"maxWidth": "500px", "margin": "40px auto", "fontFamily": "Arial, sans-serif"})

@app.callback(
    Output("result", "children"),
    Input("predict-btn", "n_clicks"),
    [State(n, "value") for n, _ in fields]
)
def do_predict(n_clicks, *values):
    if not n_clicks:
        return ""
    payload = {name: float(val) for (name,_), val in zip(fields, values)}
    try:
        r = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=10)
        r.raise_for_status()
        pred = r.json().get("prediction")
        return f"Predicted label: {pred}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)


