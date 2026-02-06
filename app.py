import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local modules
from storage import (
    init_db,
    record_reading,
    fetch_history,
    add_feedback,
    list_alerts,
    get_metrics,
)
from aqicn import fetch_aqicn
from virus_model import predict_influenza_like_protein
from virus_data import VIRUS_DB
from model import predict_risk

# Initialize Database
init_db()

AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")
if not AQICN_TOKEN:
    print("WARNING: AQICN_TOKEN is missing. External API calls will likely fail.")

CACHE_TTL_SECONDS = 600  # 10 minutes

app = FastAPI(title="AirGuard AI Backend")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache format: { "Kuala Lumpur": {"ts": 12345, "data": {...}}, "Ipoh": {...} }
_cache: Dict[str, Dict[str, Any]] = {}


# ---------------------------
# Helpers: normalization & detection
# ---------------------------

def normalize_virus_name(s: str) -> str:
    x = (s or "").strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    x = " ".join(x.split())
    aliases = {
        "influenza a": "influenza", "influenza b": "influenza", "flu a": "influenza",
        "flu b": "influenza", "h1n1": "influenza", "influenza a (h1n1)": "influenza",
        "covid-19": "covid", "covid 19": "covid", "sars-cov-2": "covid", "sars cov 2": "covid",
    }
    return aliases.get(x, x)

def looks_like_protein(raw: str) -> bool:
    if not raw: return False
    t = raw.strip()
    if not t: return False
    if t.startswith(">") or "\n" in t or "\r" in t: return True
    if " " in t: return False
    upper = t.upper()
    if not all("A" <= ch <= "Z" for ch in upper): return False
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    if any(ch not in aa for ch in upper): return False
    if len(upper) < 25: return False
    return True

def sanitize_protein_input(raw: str) -> str:
    lines = raw.splitlines()
    seq_parts = []
    for line in lines:
        t = line.strip()
        if not t or t.startswith(">"): continue
        seq_parts.append(t)
    joined = "".join(seq_parts)
    return "".join(ch for ch in joined if ch.isalpha()).upper()

def resolve_input_to_sequence(user_input: str) -> Tuple[str, Dict[str, Any]]:
    raw = (user_input or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="protein_sequence is empty")
    norm = normalize_virus_name(raw)
    if norm in VIRUS_DB:
        return VIRUS_DB[norm], {"type": "virus_name", "mapped_from": raw, "normalized_key": norm}
    if looks_like_protein(raw) or raw.startswith(">") or "\n" in raw:
        seq = sanitize_protein_input(raw)
        if not seq:
            raise HTTPException(status_code=400, detail="Empty protein sequence")
        return seq, {"type": "raw_sequence", "mapped_from": None, "normalized_key": None}
    
    supported = sorted(VIRUS_DB.keys())
    raise HTTPException(
        status_code=400,
        detail={"message": f"Unsupported virus name: '{raw}'", "hint": "Try one of the supported keys.", "supported_examples": supported[:20]}
    )

# ---------------------------
# Environment multiplier
# ---------------------------

def compute_env_multiplier(location: str = "Kuala Lumpur") -> dict:
    pts = fetch_history(location, 168)
    if not pts:
        return {"env_multiplier": 1.0, "basis": "no_history"}
    pm25_vals = [p.get("pm25") for p in pts if p.get("pm25") is not None]
    aqi_vals = [p.get("aqi") for p in pts if p.get("aqi") is not None]
    if len(pm25_vals) < 10 or len(aqi_vals) < 10:
        return {"env_multiplier": 1.0, "basis": "insufficient_history"}
    pm25_7d_mean = sum(pm25_vals) / len(pm25_vals)
    aqi_7d_mean = sum(aqi_vals) / len(aqi_vals)
    pm25_baseline = sorted(pm25_vals)[len(pm25_vals) // 2]
    if pm25_baseline <= 0: pm25_baseline = pm25_7d_mean or 1.0
    ratio = pm25_7d_mean / pm25_baseline
    alpha = 0.6
    env_multiplier = max(0.7, min(1.6, ratio ** alpha))
    return {
        "env_multiplier": float(env_multiplier),
        "basis": "pm25_7d_mean_vs_median",
        "pm25_7d_mean": float(pm25_7d_mean),
        "aqi_7d_mean": float(aqi_7d_mean),
        "pm25_baseline_median": float(pm25_baseline),
    }

# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health():
    return {"ok": True}

def get_cached_or_fetch(location: str):
    """
    Bulletproof cache retrieval.
    """
    now = time.time()
    print(f"DEBUG: Processing request for {location}") # DEBUG LOG

    if location not in _cache:
        _cache[location] = {"ts": 0, "data": None}
    
    loc_cache = _cache[location]

    # If cache is empty or old, try to fetch
    if loc_cache["data"] is None or (now - loc_cache["ts"]) > CACHE_TTL_SECONDS:
        print(f"DEBUG: Cache miss/stale for {location}. Fetching API...")
        try:
            data = fetch_aqicn(location)
        except Exception as e:
            print(f"ERROR: fetch_aqicn threw exception: {e}")
            data = None

        # --- SAFETY CHECKS START ---
        # 1. Is data None?
        # 2. Is data NOT a dictionary? (e.g. string error message)
        if data is None or not isinstance(data, dict):
            print(f"ERROR: API returned invalid data type: {type(data)} -> {data}")
            
            # Fallback to stale cache if it exists
            if loc_cache["data"]:
                print("WARN: Serving stale cache due to API failure.")
                return loc_cache["data"]
            
            # If no cache, we MUST fail safely (502 Bad Gateway)
            raise HTTPException(
                status_code=502, 
                detail=f"External API failed. Received: {str(data)[:100]}"
            )
        # --- SAFETY CHECKS END ---

        # If we get here, 'data' is definitely a Dictionary.
        data["ts"] = datetime.now(timezone.utc).isoformat()
        
        # Safely extract city name (Handle case where 'city' key is missing or not a dict)
        actual_location = location
        city_info = data.get("city")
        if isinstance(city_info, dict):
            actual_location = city_info.get("name", location)

        # Database recording (Protected)
        try:
            record_reading({
                "ts": data["ts"],
                "location": actual_location,
                "station": data.get("station"),
                "aqi": data.get("aqi"),
                "pm25": data.get("pm25"),
                "pm10": data.get("pm10"),
                "o3": data.get("o3"),
                "co": data.get("co"),
                "no2": data.get("no2"),
                "so2": data.get("so2"),
            })
        except Exception as e:
            print(f"WARN: DB write failed: {e}")

        _cache[location]["data"] = data
        _cache[location]["ts"] = now

    return _cache[location]["data"]

@app.get("/latest")
def latest(location: str = Query("Kuala Lumpur")):
    return get_cached_or_fetch(location)

class PredictRequest(BaseModel):
    protein_sequence: str
    location: str = "Kuala Lumpur"

@app.post("/predict")
def predict(payload: PredictRequest):
    user_input = (payload.protein_sequence or "").strip()
    sequence_to_use, input_info = resolve_input_to_sequence(user_input)

    data = get_cached_or_fetch(payload.location)
    
    # Double check data is valid before using
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Internal Error: Invalid AQI data structure")

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
        raise HTTPException(status_code=400, detail=f"Missing AQI/PM2.5 data for {payload.location}")

    env_pred = predict_risk(features)
    if env_pred is None:
        raise HTTPException(status_code=500, detail="Env model failed")

    try:
        p_influenza_like = predict_influenza_like_protein(sequence_to_use)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    env_info = compute_env_multiplier(payload.location)
    env_multiplier = env_info["env_multiplier"]
    overall_risk = p_influenza_like * env_multiplier

    return {
        "ts": data.get("ts"),
        "location": payload.location,
        "input_info": input_info,
        "virus_similarity": {"p_influenza_like": p_influenza_like},
        "environment": {
            "features_used": features,
            "env_prediction": env_pred,
            "env_multiplier": env_multiplier,
            "env_multiplier_details": env_info,
        },
        "overall_risk": overall_risk,
        "explanation": f"Risk calculated based on virus sequence and air quality in {payload.location}."
    }

class EnvRequest(BaseModel):
    location: str = "Kuala Lumpur"

@app.post("/predict_env")
def predict_env(payload: EnvRequest):
    data = get_cached_or_fetch(payload.location)
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
        raise HTTPException(status_code=400, detail=f"Missing AQI data for {payload.location}")

    pred = predict_risk(features)
    return {
        "ts": data.get("ts"),
        "location": payload.location,
        "features_used": features,
        "prediction": pred,
        "explanation": f"Environment-only risk for {payload.location}."
    }

@app.get("/history")
def history(location: str = Query("Kuala Lumpur"), hours: int = Query(24, ge=1, le=168)):
    return {"location": location, "hours": hours, "points": fetch_history(location, hours)}

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