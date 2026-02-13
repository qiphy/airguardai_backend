# app.py
import os
import time
import asyncio
import io
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------- Biopython Imports for NCBI --------
try:
    from Bio import Entrez, SeqIO
    
    # 1. Email is ALWAYS required by NCBI
    Entrez.email = os.environ.get("NCBI_EMAIL", "kuanqi04@gmail.com")
    
    # 2. API Key
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key
        
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("WARNING: Biopython not installed. NCBI fallback will be disabled.")

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

# -------- Restored Virus Database --------
COMMON_VIRUSES = [
    "Influenza A virus", "Influenza B virus", "SARS-CoV-2", "Dengue virus",
    "Zika virus", "Ebola virus", "Hepatitis B virus", "Hepatitis C virus",
    "Human immunodeficiency virus 1", "Herpes simplex virus 1", "Measles virus",
    "Rabies lyssavirus", "West Nile virus", "Yellow fever virus",
    "Rotavirus A", "Norovirus", "Human papillomavirus", "Chikungunya virus",
    "Plasmodium falciparum", "Mycobacterium tuberculosis"
]

# -------- Init --------
init_db()

AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")
CACHE_TTL_SECONDS = 600  # 10 minutes
_cache: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="AirGuard AI Backend")

# ---------------------------
# NCBI Helper Function
# ---------------------------
def fetch_ncbi_protein(term: str) -> str:
    if not HAS_BIOPYTHON:
        raise Exception("Biopython not installed on server")
    
    try:
        search_handle = Entrez.esearch(db="protein", term=term, retmax=1)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        id_list = search_results.get('IdList', [])
        if not id_list:
            return ""

        target_id = id_list[0]
        fetch_handle = Entrez.efetch(db="protein", id=target_id, rettype="fasta", retmode="text")
        data = fetch_handle.read()
        fetch_handle.close()

        seq_record = SeqIO.read(io.StringIO(data), "fasta")
        return str(seq_record.seq)
    except Exception as e:
        print(f"NCBI Error: {e}")
        return ""

# ---------------------------
# Virus Prediction (/predict)
# ---------------------------

class PredictPayload(BaseModel):
    location: str = "Kuala Lumpur"
    lat: float | None = None
    lng: float | None = None
    virus_name: Optional[str] = None
    protein_sequence: Optional[str] = None
    use_ncbi: bool = True

def _clean_protein(seq: str) -> str:
    lines = seq.splitlines()
    out = []
    for line in lines:
        t = line.strip()
        if not t or t.startswith(">"):
            continue
        out.append(t.upper())
    joined = "".join(out)
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join([c for c in joined if c in allowed])

@app.post("/predict")
def predict(payload: PredictPayload):
    location = (payload.location or "Kuala Lumpur").strip() or "Kuala Lumpur"

    try:
        latest = get_cached_or_fetch(location, payload.lat, payload.lng)
    except Exception as e:
        latest = _fallback_latest(location, str(e))

    features = {
        "location": location,
        "aqi": safe_float(latest.get("aqi")),
        "pm25": safe_float(latest.get("pm25")),
        "pm10": safe_float(latest.get("pm10")),
        "o3": safe_float(latest.get("o3")),
        "co": safe_float(latest.get("co")),
        "no2": safe_float(latest.get("no2")),
        "so2": safe_float(latest.get("so2")),
    }

    virus_key = None
    protein = None
    source = "unknown"

    if payload.protein_sequence and payload.protein_sequence.strip():
        protein = _clean_protein(payload.protein_sequence)
        virus_key = "custom_input"
        source = "user_input"
    elif payload.virus_name and payload.virus_name.strip():
        virus_key = payload.virus_name.strip().lower()
        if VIRUS_DB and virus_key in VIRUS_DB:
            protein = _clean_protein(VIRUS_DB[virus_key])
            source = "local_db"
        elif payload.use_ncbi and HAS_BIOPYTHON:
            raw_seq = fetch_ncbi_protein(payload.virus_name)
            if raw_seq:
                protein = _clean_protein(raw_seq)
                source = "ncbi_api"
        
        if not protein:
            raise HTTPException(status_code=404, detail=f"Virus '{payload.virus_name}' not found.")
    else:
        raise HTTPException(status_code=400, detail="Provide virus_name or protein_sequence")

    top_viruses = []
    virus_score = 0.5 
    if predict_influenza_like_protein is not None:
        try:
            out = predict_influenza_like_protein(protein)
            if isinstance(out, dict) and "top_viruses" in out:
                top_viruses = out["top_viruses"]
                if top_viruses:
                    virus_score = top_viruses[0].get("score", 0.5)
        except Exception:
            pass

    if predict_risk is None:
        pred = {"risk": "UNKNOWN", "confidence": 0.0}
    else:
        try:
            pred = predict_risk(features)
        except Exception as e:
            pred = {"risk": "UNKNOWN", "confidence": 0.0, "error": str(e)}

    blended_confidence = (pred["confidence"] * 0.6) + (virus_score * 0.4)
    
    final_risk = pred["risk"]
    if virus_score > 0.8 and final_risk == "LOW":
        final_risk = "MEDIUM"

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "location": location,
        "display_location": latest.get("display_location") or location,
        "input_type": "virus_name" if payload.virus_name else "protein_sequence",
        "virus_key": virus_key,
        "data_source": source,
        "features_used": features,
        "prediction": {
            "risk": final_risk,
            "confidence": round(blended_confidence, 4),
            "env_only_risk": pred["risk"]
        },
        "top_viruses": top_viruses,
        "aqicn_error": latest.get("error"),
    }

# -------- API Config --------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x is not None else default
    except:
        return default

def is_bad_aqi_value(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "-")

def search_aqicn(keyword: str) -> list:
    token = AQICN_TOKEN or os.environ.get("AQICN_TOKEN", "")
    if not token: return []
    try:
        url = f"https://api.waqi.info/search/?token={token}&keyword={keyword}"
        resp = requests.get(url, timeout=5)
        return resp.json().get("data", []) if resp.status_code == 200 else []
    except:
        return []

def get_cached_or_fetch(location: str, lat: float = None, lng: float = None) -> Dict[str, Any]:
    now = time.time()
    api_query = f"geo:{round(lat, 4)};{round(lng, 4)}" if lat and lng else location
    if api_query not in _cache or (now - _cache[api_query]["ts"]) > CACHE_TTL_SECONDS:
        data = fetch_aqicn(api_query)
        data["ts"] = datetime.now(timezone.utc).isoformat()
        data["display_location"] = data.get("station") or data.get("city") or location
        _cache[api_query] = {"ts": now, "data": data}
    return _cache[api_query]["data"]

def _fallback_latest(location: str, err: str) -> Dict[str, Any]:
    return {"ts": datetime.now(timezone.utc).isoformat(), "display_location": location, "error": err, "aqi": 0}

@app.get("/health")
def health(): return {"ok": True}

@app.get("/search")
def search_location(keyword: str = Query(..., min_length=2)):
    results = search_aqicn(keyword)
    return {"results": [{"name": r.get("station", {}).get("name"), "uid": r.get("uid"), "aqi": r.get("aqi")} for r in results if not is_bad_aqi_value(r.get("aqi"))]}

@app.get("/latest")
def latest(location: str = Query("Kuala Lumpur"), lat: float|None = None, lng: float|None = None):
    return get_cached_or_fetch(location, lat, lng)

@app.post("/track")
def track(payload: dict):
    uid, name = payload.get("uid"), payload.get("name", "Unknown")
    track_location(uid, name)
    return {"ok": True}

@app.post("/untrack")
def untrack(payload: dict):
    untrack_location(payload.get("uid"))
    return {"ok": True}

@app.get("/suggest")
def suggest(query: str = Query(..., min_length=2)):
    q = query.lower()
    return {"suggestions": [v for v in COMMON_VIRUSES if q in v.lower()][:5]}

@app.get("/tracked")
def tracked(): return {"tracked": list_tracked_locations()}

@app.get("/history")
def history(hours: int = 24, uid: int | None = None, location: str = "Kuala Lumpur"):
    return {"points": fetch_history_by_uid(uid, hours) if uid else fetch_history(location, hours)}

async def tracking_loop():
    while True:
        for t in list_tracked_locations():
            try:
                d = fetch_aqicn(f"@{t['uid']}")
                record_reading({**d, "ts": datetime.now(timezone.utc).isoformat(), "uid": t['uid']})
            except: pass
        await asyncio.sleep(3600)

@app.on_event("startup")
async def _startup(): asyncio.create_task(tracking_loop())

@app.post("/feedback")
def feedback(payload: dict):
    add_feedback(payload.get("name"), payload.get("profile"), payload.get("rating"), payload.get("comment"))
    return {"ok": True}

@app.get("/metrics")
def metrics(): return {"metrics": get_metrics()}

@app.get("/alerts")
def alerts(): return {"alerts": list_alerts(50)}

# ---------------------------
# Env prediction (RESTORED)
# ---------------------------

class EnvRequest(BaseModel):
    aqi: float = 0.0
    pm25: float = 0.0
    pm10: float = 0.0
    o3: float = 0.0
    co: float = 0.0
    no2: float = 0.0
    so2: float = 0.0
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
        data = {"ts": datetime.now(timezone.utc).isoformat(), "display_location": payload.location}
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
        pred = {"risk": "UNKNOWN", "confidence": 0.0, "error": "predict_risk() not available"}
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