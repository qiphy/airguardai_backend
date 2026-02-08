# app.py
import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------- Local modules --------
from storage import (
    init_db,
    record_reading,
    fetch_history,
    fetch_history_by_uid,
    track_location,
    untrack_location,
    delete_history_by_uid,
    list_tracked_locations,
    add_feedback,
    list_alerts,
    get_metrics,
)
from aqicn import fetch_aqicn

# (Optional) keep these if you have them
try:
    from model import predict_risk
except Exception:
    predict_risk = None

try:
    from virus_model import predict_influenza_like_protein
    from virus_data import VIRUS_DB
except Exception:
    predict_influenza_like_protein = None
    VIRUS_DB = {}

# -------- Init --------
init_db()

AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")
if not AQICN_TOKEN:
    print("WARNING: AQICN_TOKEN is missing. AQICN calls may fail.")

CACHE_TTL_SECONDS = 600  # 10 minutes
_cache: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="AirGuard AI Backend")

# ---------------------------
# Virus Prediction (/predict)  ✅ ADD THIS
# ---------------------------

class PredictPayload(BaseModel):
    # same key you already use everywhere
    location: str = "Kuala Lumpur"

    # one of these must be provided:
    virus_name: Optional[str] = None
    protein_sequence: Optional[str] = None


def _clean_protein(seq: str) -> str:
    # Allow FASTA headers, newlines, whitespace. Keep amino letters only.
    lines = seq.splitlines()
    out = []
    for line in lines:
        t = line.strip()
        if not t or t.startswith(">"):
            continue
        out.append(t.upper())

    joined = "".join(out)
    joined = "".join([c for c in joined if c in set("ACDEFGHIKLMNPQRSTVWY")])
    return joined


@app.post("/predict")
def predict(payload: PredictPayload):
    location = (payload.location or "Kuala Lumpur").strip() or "Kuala Lumpur"

    # -------- Name mode --------
    if payload.virus_name and payload.virus_name.strip():
        key = payload.virus_name.strip().lower()  # ✅ FIX: normalize case

        if not VIRUS_DB:
            raise HTTPException(status_code=500, detail="VIRUS_DB not available on server")

        if key not in VIRUS_DB:
            raise HTTPException(status_code=404, detail="Not Found")

        # If you have a model, you can still run it; otherwise return a simple match
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "location": location,
            "input_type": "virus_name",
            "virus_key": key,
            "top_viruses": [{"name": key, "score": 1.0}],
        }

    # -------- Protein mode --------
    if payload.protein_sequence and payload.protein_sequence.strip():
        cleaned = _clean_protein(payload.protein_sequence)
        if len(cleaned) < 25:
            raise HTTPException(status_code=400, detail="protein_sequence too short")

        if predict_influenza_like_protein is None:
            # fallback: exact match against VIRUS_DB sequences if available
            if not VIRUS_DB:
                raise HTTPException(status_code=500, detail="virus model not available and VIRUS_DB missing")

            # exact match (clean both sides)
            matches = []
            for k, seq in VIRUS_DB.items():
                if _clean_protein(seq) == cleaned:
                    matches.append({"name": k, "score": 1.0})

            if not matches:
                raise HTTPException(status_code=404, detail="Not Found")

            return {
                "ts": datetime.now(timezone.utc).isoformat(),
                "location": location,
                "input_type": "protein_sequence",
                "top_viruses": matches[:5],
                "note": "virus_model unavailable; used exact-match fallback",
            }

        # Use your actual model
        try:
            result = predict_influenza_like_protein(cleaned)
            # result can be dict or list depending on your implementation
            if isinstance(result, dict):
                return {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "location": location,
                    "input_type": "protein_sequence",
                    **result,
                }
            else:
                return {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "location": location,
                    "input_type": "protein_sequence",
                    "top_viruses": result,
                }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # -------- Neither provided --------
    raise HTTPException(status_code=400, detail="Provide virus_name or protein_sequence")


# ✅ CORS (fixes Flutter Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://airguardai.web.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:5500",
    ],
    allow_credentials=False,  # MUST be False if you ever use "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)


# ---------------------------
# Helpers
# ---------------------------

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def is_bad_aqi_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "-":
        return True
    return False


def search_aqicn(keyword: str) -> list:
    # Use AQICN search endpoint
    token = AQICN_TOKEN or os.environ.get("AQICN_TOKEN", "")
    if not token:
        return []
    try:
        url = f"https://api.waqi.info/search/?token={token}&keyword={keyword}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "ok":
                return data.get("data", [])
    except Exception as e:
        print(f"Search error: {e}")
    return []


def get_cached_or_fetch(location: str, lat: float = None, lng: float = None) -> Dict[str, Any]:
    """
    Fetch AQICN with caching.
    Supports either:
      - location text (station/city)
      - geo: lat/lng
    """
    now = time.time()

    if lat is not None and lng is not None:
        api_query = f"geo:{round(lat, 4)};{round(lng, 4)}"
        cache_key = f"geo:{round(lat, 3)},{round(lng, 3)}"
    else:
        api_query = location or "Kuala Lumpur"
        cache_key = api_query

    if cache_key not in _cache:
        _cache[cache_key] = {"ts": 0, "data": None}

    entry = _cache[cache_key]
    if entry["data"] is None or (now - entry["ts"]) > CACHE_TTL_SECONDS:
        data = fetch_aqicn(api_query)
        data["ts"] = datetime.now(timezone.utc).isoformat()

        station_name = data.get("station")
        city_name = data.get("city")
        display_location = station_name or city_name or location or api_query
        data["display_location"] = display_location

        entry["data"] = data
        entry["ts"] = now

    return entry["data"]


def _fallback_latest(location: str, err: str) -> Dict[str, Any]:
    # Never 500 to the browser
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "display_location": location,
        "error": err,
        "aqi": 0,
        "pm25": 0.0,
        "pm10": 0.0,
        "o3": 0.0,
        "co": 0.0,
        "no2": 0.0,
        "so2": 0.0,
    }


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/search")
def search_location(keyword: str = Query(..., min_length=2)):
    results = search_aqicn(keyword)

    clean = []
    for r in results:
        if is_bad_aqi_value(r.get("aqi")):
            continue
        station = r.get("station", {}) or {}
        clean.append(
            {
                "name": station.get("name", "Unknown"),
                "uid": r.get("uid"),
                "aqi": r.get("aqi"),
            }
        )
    return {"results": clean}


@app.get("/latest")
def latest(
    location: str = Query("Kuala Lumpur"),
    lat: float | None = None,
    lng: float | None = None,
):
    # ✅ Never crash -> prevents 500 + missing CORS
    try:
        return get_cached_or_fetch(location, lat, lng)
    except Exception as e:
        return _fallback_latest(location, str(e))


# ---------------------------
# Track / Untrack (UID-based)
# ---------------------------

class TrackRequest(BaseModel):
    uid: int
    name: str

@app.post("/track")
def track(payload: TrackRequest):
    uid = int(payload.uid)
    if uid <= 0:
        raise HTTPException(status_code=400, detail="uid is required")

    name = (payload.name or "").strip() or f"@{uid}"

    # save tracked
    track_location(uid, name)

    # ✅ record immediately so Trends has points right away
    try:
        data = fetch_aqicn(city=f"@{uid}")
        ts = datetime.now(timezone.utc).isoformat()

        station_name = data.get("station") or name
        display_location = station_name or name

        record_reading(
            {
                "ts": ts,
                "uid": uid,
                "location": display_location,
                "station": station_name,
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
        print(f"WARN: /track immediate record failed uid={uid}: {e}")

    return {"ok": True, "tracked": {"uid": uid, "name": name}}


class UntrackRequest(BaseModel):
    uid: int

@app.post("/untrack")
def untrack(payload: UntrackRequest):
    uid = int(payload.uid)
    if uid <= 0:
        raise HTTPException(status_code=400, detail="uid is required")

    delete_history_by_uid(uid)
    untrack_location(uid)
    return {"ok": True, "removed_uid": uid}


@app.get("/tracked")
def tracked():
    return {"tracked": list_tracked_locations()}


# ---------------------------
# History
# ---------------------------

@app.get("/history")
def history(
    hours: int = Query(24, ge=1, le=168),
    uid: int | None = None,
    location: str = Query("Kuala Lumpur"),
):
    if uid is not None:
        return {"uid": uid, "hours": hours, "points": fetch_history_by_uid(uid, hours)}
    return {"location": location, "hours": hours, "points": fetch_history(location, hours)}


# ---------------------------
# Env prediction (so Flutter won't 404)
# ---------------------------

class EnvRequest(BaseModel):
    # Direct inputs (preferred)
    aqi: float = 0.0
    pm25: float = 0.0
    pm10: float = 0.0
    o3: float = 0.0
    co: float = 0.0
    no2: float = 0.0
    so2: float = 0.0

    # optional fallback lookup
    location: str = "Kuala Lumpur"
    lat: float | None = None
    lng: float | None = None

@app.post("/predict_env")
def predict_env(payload: EnvRequest):
    has_direct = any([
        payload.aqi != 0.0,
        payload.pm25 != 0.0,
        payload.pm10 != 0.0,
        payload.o3 != 0.0,
        payload.co != 0.0,
        payload.no2 != 0.0,
        payload.so2 != 0.0,
    ])

    if has_direct:
        data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "display_location": payload.location,
        }
        features = {
            "location": payload.location,
            "aqi": safe_float(payload.aqi),
            "pm25": safe_float(payload.pm25),
            "pm10": safe_float(payload.pm10),
            "o3": safe_float(payload.o3),
            "co": safe_float(payload.co),
            "no2": safe_float(payload.no2),
            "so2": safe_float(payload.so2),
        }
    else:
        try:
            data = get_cached_or_fetch(payload.location, payload.lat, payload.lng)
        except Exception as e:
            data = _fallback_latest(payload.location, str(e))

        features = {
            "location": payload.location,
            "aqi": safe_float(data.get("aqi")),
            "pm25": safe_float(data.get("pm25")),
            "pm10": safe_float(data.get("pm10")),
            "o3": safe_float(data.get("o3")),
            "co": safe_float(data.get("co")),
            "no2": safe_float(data.get("no2")),
            "so2": safe_float(data.get("so2")),
        }

    if predict_risk is None:
        pred = {"risk": "UNKNOWN", "confidence": 0.0, "error": "predict_risk() not available on server"}
    else:
        try:
            pred = predict_risk(features)
        except Exception as e:
            pred = {"risk": "UNKNOWN", "confidence": 0.0, "error": str(e)}

    return {
        "ts": data.get("ts"),
        "location": payload.location,
        "display_location": data.get("display_location") or payload.location,
        "features_used": features,
        "prediction": pred,
        "aqicn_error": data.get("error"),
        "used_direct_inputs": has_direct,
    }


# ---------------------------
# Optional: background hourly tracker (keeps history growing)
# ---------------------------

async def tracking_loop():
    while True:
        try:
            tracked_list = list_tracked_locations()
            for t in tracked_list:
                uid = int(t["uid"])
                name = (t.get("name") or "").strip() or f"@{uid}"
                try:
                    data = fetch_aqicn(city=f"@{uid}")
                    ts = datetime.now(timezone.utc).isoformat()

                    station_name = data.get("station") or name
                    display_location = station_name or name

                    record_reading(
                        {
                            "ts": ts,
                            "uid": uid,
                            "location": display_location,
                            "station": station_name,
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
                    print(f"WARN: tracking_loop failed uid={uid}: {e}")
        except Exception as e:
            print(f"WARN: tracking_loop outer error: {e}")

        await asyncio.sleep(3600)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(tracking_loop())


# ---------------------------
# Existing misc endpoints
# ---------------------------

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
