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
    
    # 2. API Key (Optional but recommended for higher limits)
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

# -------- Init --------
init_db()

AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")
if not AQICN_TOKEN:
    print("WARNING: AQICN_TOKEN is missing. AQICN calls may fail.")

CACHE_TTL_SECONDS = 600  # 10 minutes
_cache: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="AirGuard AI Backend")

# ✅ CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev; restrict in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# NCBI Helper Functions
# ---------------------------

@app.get("/suggest")
def suggest(query: str = Query(..., min_length=2)):
    """
    Proxies NCBI's Clinical Tables API for fast organism/virus name autocomplete.
    Includes timeout and headers to prevent crashing on rapid requests.
    """
    try:
        url = "https://clinicaltables.nlm.nih.gov/api/organism/v3/search"
        params = {
            "terms": query,
            "maxList": 7,
            "df": "species"
        }
        # Identifying User-Agent prevents blocking
        headers = {
            "User-Agent": "AirGuardAI/1.0 (contact@example.com)"
        }
        
        # Timeout set to 5s to prevent hanging threads
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            if len(data) >= 4:
                return {"suggestions": data[3]}
                
    except requests.exceptions.Timeout:
        print("Autocomplete: Timeout")
    except Exception as e:
        print(f"Autocomplete Error: {e}")
        
    # Return empty list on failure instead of crashing
    return {"suggestions": []}


def fetch_ncbi_protein(term: str) -> str:
    if not HAS_BIOPYTHON:
        raise Exception("Biopython not installed on server")

    print(f"NCBI: Searching for '{term}'...")
    try:
        search_handle = Entrez.esearch(db="protein", term=term, retmax=1)
        search_results = Entrez.read(search_handle)
        search_handle.close()
    except Exception as e:
        print(f"NCBI Search Failed: {e}")
        return ""

    id_list = search_results.get('IdList', [])
    if not id_list:
        return ""

    target_id = id_list[0]
    print(f"NCBI: Found ID {target_id}, fetching...")

    try:
        fetch_handle = Entrez.efetch(db="protein", id=target_id, rettype="fasta", retmode="text")
        data = fetch_handle.read()
        fetch_handle.close()
        seq_record = SeqIO.read(io.StringIO(data), "fasta")
        return str(seq_record.seq)
    except Exception as e:
        print(f"NCBI Fetch Failed: {e}")
        return ""


# ---------------------------
# Virus Prediction (/predict)
# ---------------------------

class PredictPayload(BaseModel):
    location: str = "Unknown"
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
    joined = "".join([c for c in joined if c in allowed])
    return joined


@app.post("/predict_env")
def predict(payload: PredictPayload):
    location = (payload.location or "Unknown").strip() or "Unknown"

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
        if len(protein) < 10:
            raise HTTPException(status_code=400, detail="protein_sequence too short")
        virus_key = "custom_input"
        source = "user_input"

    elif payload.virus_name and payload.virus_name.strip():
        virus_key = payload.virus_name.strip().lower()

        if VIRUS_DB and virus_key in VIRUS_DB:
            protein = _clean_protein(VIRUS_DB[virus_key])
            source = "local_db"
        
        elif payload.use_ncbi and HAS_BIOPYTHON:
            try:
                raw_seq = fetch_ncbi_protein(payload.virus_name)
                if raw_seq:
                    protein = _clean_protein(raw_seq)
                    source = "ncbi_api"
            except Exception as e:
                print(f"NCBI error: {e}")
        
        if not protein:
            raise HTTPException(status_code=404, detail=f"Virus '{payload.virus_name}' not found in local DB or NCBI.")

    else:
        raise HTTPException(status_code=400, detail="Provide virus_name or protein_sequence")

    top_viruses = []
    if predict_influenza_like_protein is not None:
        try:
            out = predict_influenza_like_protein(protein)
            if isinstance(out, dict) and "top_viruses" in out:
                top_viruses = out["top_viruses"]
            elif isinstance(out, list):
                top_viruses = out
        except Exception as e:
            top_viruses = [{"name": virus_key, "score": 1.0, "error": str(e)}]
    else:
        top_viruses = [{"name": virus_key, "score": 1.0, "note": "Model not loaded"}]

    if predict_risk is None:
        pred = {"risk": "UNKNOWN", "confidence": 0.0, "error": "predict_risk() not available"}
    else:
        try:
            pred = predict_risk(features)
        except Exception as e:
            pred = {"risk": "UNKNOWN", "confidence": 0.0, "error": str(e)}

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "location": location,
        "display_location": latest.get("display_location") or location,
        "input_type": "virus_name" if payload.virus_name else "protein_sequence",
        "virus_key": virus_key,
        "data_source": source,
        "features_used": features,
        "prediction": pred,
        "top_viruses": top_viruses,
        "aqicn_error": latest.get("error"),
    }

# ---------------------------
# Helpers & Other Routes
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
    now = time.time()
    if lat is not None and lng is not None:
        api_query = f"geo:{round(lat, 4)};{round(lng, 4)}"
        cache_key = f"geo:{round(lat, 3)},{round(lng, 3)}"
    else:
        api_query = location or "Unknown"
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
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "display_location": location,
        "error": err,
        "aqi": 0, "pm25": 0.0, "pm10": 0.0, "o3": 0.0,
        "co": 0.0, "no2": 0.0, "so2": 0.0,
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/latest")
def latest(location: str = Query("Unknown"), lat: float | None = None, lng: float | None = None):
    try:
        return get_cached_or_fetch(location, lat, lng)
    except Exception as e:
        return _fallback_latest(location, str(e))

class TrackRequest(BaseModel):
    uid: int
    name: str

@app.post("/track")
def track(payload: TrackRequest):
    uid = int(payload.uid)
    name = (payload.name or "").strip() or f"@{uid}"
    track_location(uid, name)
    try:
        data = fetch_aqicn(city=f"@{uid}")
        record_reading({
            "ts": datetime.now(timezone.utc).isoformat(),
            "uid": uid,
            "location": data.get("station") or name,
            "station": data.get("station") or name,
            "aqi": data.get("aqi"),
            "pm25": data.get("pm25"), "pm10": data.get("pm10"),
            "o3": data.get("o3"), "co": data.get("co"),
            "no2": data.get("no2"), "so2": data.get("so2"),
        })
    except Exception:
        pass
    return {"ok": True}

class UntrackRequest(BaseModel):
    uid: int

@app.post("/untrack")
def untrack(payload: UntrackRequest):
    delete_history_by_uid(payload.uid)
    untrack_location(payload.uid)
    return {"ok": True}

@app.get("/tracked")
def tracked():
    return {"tracked": list_tracked_locations()}

@app.get("/history")
def history(hours: int = Query(24), uid: int | None = None, location: str = Query("Unknown")):
    if uid is not None:
        return {"uid": uid, "hours": hours, "points": fetch_history_by_uid(uid, hours)}
    return {"location": location, "hours": hours, "points": fetch_history(location, hours)}

# Background tracking loop
async def tracking_loop():
    while True:
        try:
            tracked_list = list_tracked_locations()
            for t in tracked_list:
                try:
                    uid = int(t["uid"])
                    data = fetch_aqicn(city=f"@{uid}")
                    record_reading({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "uid": uid,
                        "location": data.get("station"),
                        "station": data.get("station"),
                        "aqi": data.get("aqi"),
                        "pm25": data.get("pm25"),
                    })
                except Exception:
                    pass
        except Exception:
            pass
        await asyncio.sleep(3600)

@app.on_event("startup")
async def _startup():
    asyncio.create_task(tracking_loop())