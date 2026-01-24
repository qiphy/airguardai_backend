import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from storage import init_db, record_reading, fetch_history, add_feedback, add_alert, list_alerts, get_metrics
init_db()
from datetime import datetime, timezone
from fastapi import Query
from pydantic import BaseModel
from aqicn import fetch_aqicn
from virus_model import predict_influenza_like_protein


# Local ML model (scikit-learn)
from model import predict_risk

def compute_env_multiplier(location: str = "Kuala Lumpur") -> dict:
    # Use the last 7 days (168h) from your real stored AQICN pulls
    pts = fetch_history(location, 168)
    if not pts:
        # fallback if DB is empty
        return {"env_multiplier": 1.0, "basis": "no_history"}

    pm25_vals = [p.get("pm25") for p in pts if p.get("pm25") is not None]
    aqi_vals = [p.get("aqi") for p in pts if p.get("aqi") is not None]

    if len(pm25_vals) < 10 or len(aqi_vals) < 10:
        return {"env_multiplier": 1.0, "basis": "insufficient_history"}

    pm25_7d_mean = sum(pm25_vals) / len(pm25_vals)
    aqi_7d_mean = sum(aqi_vals) / len(aqi_vals)

    # Stable, conservative multiplier (hackathon-friendly):
    # baseline is "typical" KL air quality from the same window
    pm25_baseline = sorted(pm25_vals)[len(pm25_vals)//2]  # median
    if pm25_baseline <= 0:
        pm25_baseline = pm25_7d_mean or 1.0

    ratio = pm25_7d_mean / pm25_baseline

    # sublinear scaling + clamp (avoid crazy spikes)
    alpha = 0.6
    env_multiplier = max(0.7, min(1.6, ratio ** alpha))

    return {
        "env_multiplier": float(env_multiplier),
        "basis": "pm25_7d_mean_vs_median",
        "pm25_7d_mean": float(pm25_7d_mean),
        "aqi_7d_mean": float(aqi_7d_mean),
        "pm25_baseline_median": float(pm25_baseline),
    }


# IMPORTANT: Set AQICN_TOKEN in your environment.
# macOS/Linux:  export AQICN_TOKEN="..."
# Windows PS:   $env:AQICN_TOKEN="..."
AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")
LOCATION = "kuala%20lumpur"
CACHE_TTL_SECONDS = 600  # 10 minutes

app = FastAPI(title="AirGuard AI Local Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {
    "ts": 0,
    "data": None
}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/latest")
def latest():
    now = time.time()
    if _cache["data"] is None or (now - _cache["ts"]) > CACHE_TTL_SECONDS:
        data = fetch_aqicn()
        data["ts"] = datetime.now(timezone.utc).isoformat()
        record_reading({
            "ts": data["ts"],
            "location": "Kuala Lumpur",
            "station": data.get("station"),
            "aqi": data.get("aqi"),
            "pm25": data.get("pm25"),
            "pm10": data.get("pm10"),
            "o3": data.get("o3"),
            "co": data.get("co"),
            "no2": data.get("no2"),
            "so2": data.get("so2"),
        })
        _cache["data"] = data
        _cache["ts"] = now
    return _cache["data"]

class PredictRequest(BaseModel):
    protein_sequence: str
    location: str = "Kuala Lumpur"

@app.post("/predict_env")
def predict_env():
    data = _cache.get("data")
    if data is None:
        data = latest()

    features = {
        "location": "Kuala Lumpur",
        "aqi": data.get("aqi"),
        "pm25": data.get("pm25"),
        "pm10": data.get("pm10"),
        "o3": data.get("o3"),
        "co": data.get("co"),
        "no2": data.get("no2"),
        "so2": data.get("so2"),
    }
    if features["aqi"] is None or features["pm25"] is None:
        raise HTTPException(status_code=400, detail=f"Missing fields for prediction: {features}")

    pred = predict_risk(features)
    return {
        "ts": data.get("ts"),
        "location": "Kuala Lumpur",
        "features_used": features,
        "prediction": pred,
        "explanation": "Environment-only risk from AQI/PM2.5.",
    }

@app.post("/predict")
def predict(payload: PredictRequest):
    # 1) Ensure we have latest AQI data
    data = _cache.get("data")
    if data is None:
        data = latest()

    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Latest AQI data is not available")

    # 2) Environment features (same as you already do)
    features = {
        "location": payload.location,
        "aqi": data.get("aqi"),
        "pm25": data.get("pm25"),
        "pm10": data.get("pm10"),
        "o3": data.get("o3"),
        "co": data.get("co"),
        "no2": data.get("no2"),
        "so2": data.get("so2"),
    }

    if features["aqi"] is None or features["pm25"] is None:
        raise HTTPException(status_code=400, detail=f"Missing fields for prediction: {features}")

    env_pred = predict_risk(features)
    if env_pred is None or not isinstance(env_pred, dict):
        raise HTTPException(status_code=500, detail="Env model returned invalid prediction")

    # 3) Virus similarity from protein sequence
    try:
        p_influenza_like = predict_influenza_like_protein(payload.protein_sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 4) Compute environment multiplier from REAL history
    env_info = compute_env_multiplier(payload.location)
    env_multiplier = env_info["env_multiplier"]

    # 5) Combine
    overall_risk = p_influenza_like * env_multiplier

    return {
        "ts": data.get("ts"),
        "location": payload.location,
        "virus_similarity": {
            "p_influenza_like": p_influenza_like
        },
        "environment": {
            "features_used": features,
            "env_prediction": env_pred,
            "env_multiplier": env_multiplier,
            "env_multiplier_details": env_info,
        },
        "overall_risk": overall_risk,
        "explanation": (
            "Virus similarity is predicted from the submitted protein sequence only. "
            "Air quality (AQI/PM2.5) is used only to modulate environmental susceptibility via a multiplier."
        )
    }


@app.get("/history")
def history(hours: int = Query(24, ge=1, le=168)):
    return {"location": "Kuala Lumpur", "hours": hours, "points": fetch_history("Kuala Lumpur", hours)}

class FeedbackIn(BaseModel):
    name: str = "Anonymous"
    profile: str = "General"
    rating: int
    comment: str = ""

@app.post("/feedback")
def feedback(payload: FeedbackIn):
    add_feedback(payload.name, payload.profile, payload.rating, payload.comment)
    return {"ok": True}

@app.get("/metrics")
def metrics():
    return {"ok": True, "metrics": get_metrics()}

@app.get("/alerts")
def alerts():
    return {"alerts": list_alerts(50)}