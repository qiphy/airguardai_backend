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
CACHE_TTL_SECONDS = 600  # 10 minutes

app = FastAPI(title="AirGuard AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {"ts": 0, "data": None}


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
    """
    Heuristic to decide whether the input is a protein sequence (or FASTA),
    versus a virus name.

    - FASTA headers / multiline -> sequence
    - Spaces -> likely name
    - Only allow standard 20 AAs (ACDEFGHIKLMNPQRSTVWY)
    - Very short strings are treated as names
    """
    if not raw:
        return False

    t = raw.strip()
    if not t:
        return False

    # FASTA or multiline: treat as sequence (we'll sanitize elsewhere if needed)
    if t.startswith(">") or "\n" in t or "\r" in t:
        return True

    # Names usually contain spaces, slashes, parentheses, etc.
    if " " in t:
        return False

    upper = t.upper()

    # Allow only letters to be considered a candidate sequence
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
    """
    - Removes FASTA headers & whitespace
    - Keeps letters only
    - Uppercases
    """
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
    """
    Returns (sequence_to_use, input_info)

    Behavior:
    - If input matches VIRUS_DB key after normalization -> map to stored sequence.
    - Else, if it looks like a protein -> sanitize and use as protein.
    - Else -> reject as unsupported virus name (prevents confusing model errors).
    """
    raw = (user_input or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="protein_sequence is empty")

    # Try virus-name path first (normalized)
    norm = normalize_virus_name(raw)
    if norm in VIRUS_DB:
        return VIRUS_DB[norm], {
            "type": "virus_name",
            "mapped_from": raw,
            "normalized_key": norm,
        }

    # Then try protein path
    if looks_like_protein(raw) or raw.startswith(">") or "\n" in raw or "\r" in raw:
        seq = sanitize_protein_input(raw)
        if not seq:
            raise HTTPException(status_code=400, detail="Protein sequence is empty after sanitization")
        return seq, {
            "type": "raw_sequence",
            "mapped_from": None,
            "normalized_key": None,
        }

    # Otherwise it's a name we don't support
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
    """
    Calculates a risk multiplier based on the last 7 days of air quality data.
    """
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


@app.get("/latest")
def latest():
    now = time.time()

    if _cache["data"] is None or (now - _cache["ts"]) > CACHE_TTL_SECONDS:
        data = fetch_aqicn()
        data["ts"] = datetime.now(timezone.utc).isoformat()

        record_reading(
            {
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
            }
        )

        _cache["data"] = data
        _cache["ts"] = now

    return _cache["data"]


class PredictRequest(BaseModel):
    protein_sequence: str
    location: str = "Kuala Lumpur"


@app.post("/predict")
def predict(payload: PredictRequest):
    # ---------------------------------------------------------
    # STEP 1: Resolve input to a protein sequence
    # ---------------------------------------------------------
    user_input = (payload.protein_sequence or "").strip()
    sequence_to_use, input_info = resolve_input_to_sequence(user_input)

    # ---------------------------------------------------------
    # STEP 2: Get Air Quality Data
    # ---------------------------------------------------------
    data = _cache.get("data")
    if data is None:
        data = latest()

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

    # ---------------------------------------------------------
    # STEP 3: Run Models
    # ---------------------------------------------------------
    env_pred = predict_risk(features)
    if env_pred is None:
        raise HTTPException(status_code=500, detail="Env model returned invalid prediction")

    try:
        p_influenza_like = predict_influenza_like_protein(sequence_to_use)
    except ValueError as e:
        # Keep as 400: user input issue (bad sequence)
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
            "Virus similarity is predicted from the submitted protein sequence (or a mapped virus name). "
            "Air quality (AQI/PM2.5) modulates susceptibility via a multiplier."
        ),
    }


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
