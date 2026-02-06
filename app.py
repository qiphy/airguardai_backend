import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
import requests 

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

# Cache format: { "Kuala Lumpur": {"ts": 12345, "data": {...}}, "geo:3.1,101.5": {...} }
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

# Search helper function for autocomplete
def search_aqicn(keyword: str) -> list:
    if not AQICN_TOKEN:
        return []
    try:
        url = f"https://api.waqi.info/search/?token={AQICN_TOKEN}&keyword={keyword}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "ok":
                return data.get("data", [])
    except Exception as e:
        print(f"Search error: {e}")
    return []

@app.get("/search")
def search_location(keyword: str = Query(..., min_length=2)):
    results = search_aqicn(keyword)
    clean_results = []
    for r in results:
        station = r.get("station", {})
        clean_results.append({
            "name": station.get("name", "Unknown"),
            "uid": r.get("uid"),
            "aqi": r.get("aqi")
        })
    return {"results": clean_results}

# ---------------------------------------------------------
# UPDATED: Robust Cache & Fetch Logic (Fixes Unknown Station)
# ---------------------------------------------------------

def get_cached_or_fetch(location: str, lat: float = None, lng: float = None):
    now = time.time()
    
    # 1. Determine Query with Rounding (Fixes API precision error)
    if lat is not None and lng is not None:
        # Round to 4 decimals max for API query
        api_query = f"geo:{round(lat, 4)};{round(lng, 4)}"
        # Cache key rounded to 3 decimals to group nearby users
        cache_key = f"geo:{round(lat, 3)},{round(lng, 3)}"
    else:
        api_query = location
        cache_key = location

    print(f"DEBUG: Processing request for {cache_key} (Query: {api_query})") 

    if cache_key not in _cache:
        _cache[cache_key] = {"ts": 0, "data": None}
    
    loc_cache = _cache[cache_key]

    # 2. Fetch if stale
    if loc_cache["data"] is None or (now - loc_cache["ts"]) > CACHE_TTL_SECONDS:
        data = None
        try:
            print(f"DEBUG: Fetching AQICN for {api_query}...")
            data = fetch_aqicn(api_query)
        except Exception as e:
            print(f"ERROR: fetch_aqicn error: {e}")

        # 3. SMART FALLBACK
        # If Geo-lookup failed (data is None) AND we have a valid city name, try the name.
        # But ignore "Current Location" because that's not a city name.
        if (data is None) and (lat is not None) and location and (location != "Current Location"):
            print(f"WARN: Geo-lookup failed. Falling back to text search: '{location}'")
            try:
                data = fetch_aqicn(location)
            except Exception as e:
                print(f"ERROR: Fallback fetch failed: {e}")

        # 4. Final Validation
        if data is None or not isinstance(data, dict):
            # If we have OLD cache, serve it (Better than crashing)
            if loc_cache["data"]:
                print("WARN: Serving stale cache due to API failure.")
                return loc_cache["data"]
            
            # If no data at all, return 502
            print(f"CRITICAL: API failed for {api_query} and no fallback available.")
            raise HTTPException(
                status_code=502, 
                detail="External API failed to find station for this location."
            )

        # 5. Success - Update Cache & DB
        data["ts"] = datetime.now(timezone.utc).isoformat()
        
        # Use the real station name returned by API
        actual_location = data.get("city", {}).get("name", location)

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

        _cache[cache_key]["data"] = data
        _cache[cache_key]["ts"] = now

    return _cache[cache_key]["data"]

@app.get("/latest")
def latest(location: str = Query("Kuala Lumpur"), lat: float = None, lng: float = None):
    return get_cached_or_fetch(location, lat, lng)

class PredictRequest(BaseModel):
    protein_sequence: str
    location: str = "Kuala Lumpur"

@app.post("/predict")
def predict(payload: PredictRequest):
    user_input = (payload.protein_sequence or "").strip()
    sequence_to_use, input_info = resolve_input_to_sequence(user_input)

    # Note: Predict doesn't usually carry coords, so we use location string
    data = get_cached_or_fetch(payload.location)
    
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
        pass # Allow models to handle missing data or defaults in predict_risk

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