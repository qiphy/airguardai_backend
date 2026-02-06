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

# FIXED: CORS setup for broad compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # Must be False for wildcard '*' origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache format: { "Kuala Lumpur": {"ts": 12345, "data": {...}}, "Ipoh": {...} }
_cache: Dict[str, Dict[str, Any]] = {}


# ---------------------------
# Helpers: normalization & detection
# ---------------------------

def normalize_virus_name(s: str) -> str:
    """
    Normalize user-entered virus names into keys compatible with VIRUS_DB.
    """
    x = (s or "").strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    x = " ".join(x.split())

    aliases = {
        # influenza variants
        "influenza a": "influenza",
        "influenza b": "influenza",
        "flu a": "influenza",
        "flu b": "influenza",
        "h1n1": "influenza",
        "influenza a (h1n1)": "influenza",

        # covid variants
        "covid-19": "covid",
        "covid 19": "covid",
        "sars-cov-2": "covid",
        "sars cov 2": "covid",
    }
    return aliases.get(x, x)


def looks_like_protein(raw: str) -> bool:
    if not raw:
        return False
    t = raw.strip()
    if not t:
        return False
    # FASTA or multiline
    if t.startswith(">") or "\n" in t or "\r" in t:
        return True
    # Names usually contain spaces
    if " " in t:
        return False
    upper = t.upper()
    if not all("A" <= ch <= "Z" for ch in upper):
        return False
    # Standard 20 amino acids only
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    if any(ch not in aa for ch in upper):
        return False
    # Too short is likely a label ("flu", "covid")
    if len(upper) < 25:
        return False
    return True


def sanitize_protein_input(raw: str) -> str:
    lines = raw.splitlines()
    seq_parts = []
    for line in lines:
        t = line.strip()
        if not t:
            continue
        if t.startswith(">"):
            continue
        seq_parts.append(t)
    joined = "".join(seq_parts)
    cleaned = "".join(ch for ch in joined if ch.isalpha()).upper()
    return cleaned


def resolve_input_to_sequence(user_input: str) -> Tuple[str, Dict[str, Any]]:
    raw = (user_input or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="protein_sequence is empty")

    norm = normalize_virus_name(raw)
    if norm in VIRUS_DB:
        return VIRUS_DB[norm], {
            "type": "virus_name",
            "mapped_from": raw,
            "normalized_key": norm,
        }

    if looks_like_protein(raw) or raw.startswith(">") or "\n" in raw or "\r" in raw:
        seq = sanitize_protein_input(raw)
        if not seq:
            raise HTTPException(status_code=400, detail="Protein sequence is empty after sanitization")
        return seq, {
            "type": "raw_sequence",
            "mapped_from": None,
            "normalized_key": None,
        }

    supported = sorted(VIRUS_DB.keys())
    raise HTTPException(
        status_code=400,
        detail={
            "message": f"Unsupported virus name: '{raw}'",
            "hint": "Try one of the supported keys (or paste a protein sequence/FASTA).",
            "supported_examples": supported[:20],
        },
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
    if pm25_baseline <= 0:
        pm25_baseline = pm25_7d_mean or 1.0

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
    Robust cache retrieval that handles API failures and bad return types gracefully.
    """
    now = time.time()
    
    # Initialize cache for this location if it doesn't exist
    if location not in _cache:
        _cache[location] = {"ts": 0, "data": None}
    
    loc_cache = _cache[location]

    # Check if cache is stale or empty
    if loc_cache["data"] is None or (now - loc_cache["ts"]) > CACHE_TTL_SECONDS:
        try:
            # Pass location to your fetch function
            data = fetch_aqicn(location) 
        except Exception as e:
            print(f"Error calling fetch_aqicn for {location}: {e}")
            data = None

        # CRITICAL FIX 1: Ensure data is not None
        # CRITICAL FIX 2: Ensure data is actually a Dict (prevents 'str' object has no attribute 'get')
        if not data or not isinstance(data, dict):
            print(f"API returned invalid data for {location}: {data}")
            
            # If we have old cache, return it (stale is better than crashing)
            if loc_cache["data"]:
                print("Serving stale cache.")
                return loc_cache["data"]
            
            # If no cache and API failed, we must error out
            raise HTTPException(
                status_code=502, 
                detail=f"Unable to fetch air quality data for '{location}'. API returned: {data}"
            )

        # Proceed if data is valid
        data["ts"] = datetime.now(timezone.utc).isoformat()
        
        # Ensure the data returned actually has the location name, or fallback to requested
        actual_location = data.get("city", {}).get("name", location)

        # Wrap DB write in try/except so DB issues don't block the API response
        try:
            record_reading(
                {
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
                }
            )
        except Exception as e:
            print(f"Warning: Failed to record reading to DB: {e}")

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
    # 1. Resolve Sequence
    user_input = (payload.protein_sequence or "").strip()
    sequence_to_use, input_info = resolve_input_to_sequence(user_input)

    # 2. Get Air Quality Data (Using the helper for dynamic location)
    data = get_cached_or_fetch(payload.location)

    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Latest AQI data is not available")

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

    # 3. Run Models
    env_pred = predict_risk(features)
    if env_pred is None:
        raise HTTPException(status_code=500, detail="Env model returned invalid prediction")

    try:
        p_influenza_like = predict_influenza_like_protein(sequence_to_use)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        "explanation": (
            "Virus similarity is predicted from the submitted protein sequence. "
            f"Air quality context ({payload.location}) modulates susceptibility."
        ),
    }


class EnvRequest(BaseModel):
    location: str = "Kuala Lumpur"


@app.post("/predict_env")
def predict_env(payload: EnvRequest):
    # Use helper to get data for specific location
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
        raise HTTPException(status_code=400, detail=f"Missing fields for prediction: {features}")

    pred = predict_risk(features)
    return {
        "ts": data.get("ts"),
        "location": payload.location,
        "features_used": features,
        "prediction": pred,
        "explanation": f"Environment-only risk from AQI/PM2.5 for {payload.location}.",
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