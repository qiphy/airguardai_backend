import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

init_db()

AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")
CACHE_TTL_SECONDS = 600

app = FastAPI(title="AirGuard AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Dict[str, Dict[str, Any]] = {}


def is_bad_aqi_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "-":
        return True
    return False


@app.get("/health")
def health():
    return {"ok": True}


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
        if is_bad_aqi_value(r.get("aqi")):
            continue
        station = r.get("station", {}) or {}
        clean_results.append(
            {
                "name": station.get("name", "Unknown"),
                "uid": r.get("uid"),
                "aqi": r.get("aqi"),
            }
        )
    return {"results": clean_results}


# ---------------------------
# Track / Untrack
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

    # ✅ record immediately so charts show instantly
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

    # delete old history + stop tracking
    delete_history_by_uid(uid)
    untrack_location(uid)
    return {"ok": True, "removed_uid": uid}


@app.get("/tracked")
def tracked():
    return {"tracked": list_tracked_locations()}


# ---------------------------
# Latest (optional cache)
# ---------------------------

def get_cached_or_fetch(location: str):
    now = time.time()
    cache_key = location

    if cache_key not in _cache:
        _cache[cache_key] = {"ts": 0, "data": None}

    loc_cache = _cache[cache_key]

    if loc_cache["data"] is None or (now - loc_cache["ts"]) > CACHE_TTL_SECONDS:
        data = fetch_aqicn(location)
        data["ts"] = datetime.now(timezone.utc).isoformat()
        loc_cache["data"] = data
        loc_cache["ts"] = now

    return loc_cache["data"]


@app.get("/latest")
def latest(location: str = Query("Kuala Lumpur")):
    return get_cached_or_fetch(location)


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
# Background hourly tracking (optional; keeps building history)
# ---------------------------

async def tracking_loop():
    while True:
        try:
            tracked_list = list_tracked_locations()
            for t in tracked_list:
                uid = int(t["uid"])
                name = t.get("name") or f"@{uid}"
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
# Existing endpoints you already had (kept minimal)
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
