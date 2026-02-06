from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional, List

import requests

AQICN_BASE = "https://api.waqi.info"

class AQICNError(RuntimeError):
    pass

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _get_iaqi_value(iaqi: Dict[str, Any], key: str) -> Optional[float]:
    obj = iaqi.get(key)
    if not isinstance(obj, dict):
        return None
    return _to_float(obj.get("v"))

def _get_forecast_value(forecast: Dict[str, Any], key: str, target_date: str) -> Optional[float]:
    daily_data = forecast.get("daily", {}).get(key, [])
    if not isinstance(daily_data, list):
        return None
    for entry in daily_data:
        if entry.get("day") == target_date:
            return _to_float(entry.get("max"))
    return None

def fetch_aqicn(
    city: str = "Kuala Lumpur",
    token: Optional[str] = None,
    timeout: int = 12,
) -> Dict[str, Any]:
    token = token or os.getenv("AQICN_TOKEN")
    if not token:
        # This error will be caught by app.py and shown in logs
        raise AQICNError("Missing AQICN_TOKEN environment variable.")

    # Handles both "Kuala Lumpur" and "geo:3.1;101.5"
    url = f"{AQICN_BASE}/feed/{city}/"
    params = {"token": token}

    try:
        r = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise AQICNError(f"Network error calling AQICN: {e}") from e

    if r.status_code != 200:
        raise AQICNError(f"AQICN HTTP {r.status_code}: {r.text}")

    try:
        raw = r.json()
    except Exception as e:
        raise AQICNError(f"AQICN returned non-JSON response: {r.text[:500]}") from e

    status = raw.get("status")
    if status != "ok":
        # Pass the message up so app.py can decide to fallback or fail
        raise AQICNError(f"AQICN status={status}, data={raw.get('data')}")

    d = raw.get("data") or {}
    iaqi = d.get("iaqi") or {}
    forecast = d.get("forecast") or {}

    # 1. Determine Date for Forecast
    api_time_str = d.get("time", {}).get("s", "")
    current_date_str = None
    if api_time_str:
        try:
            current_date_str = api_time_str.split(" ")[0] 
        except Exception:
            pass
    if not current_date_str:
        current_date_str = datetime.utcnow().strftime("%Y-%m-%d")

    # 2. Extract Info
    station = None
    city_obj = d.get("city")
    if isinstance(city_obj, dict):
        station = city_obj.get("name")

    aqi = _to_float(d.get("aqi"))

    # 3. Helper: Realtime -> Forecast -> Default 0.0
    def get_val(pol_key: str) -> float:
        val = _get_iaqi_value(iaqi, pol_key)
        if val is None and current_date_str:
            val = _get_forecast_value(forecast, pol_key, current_date_str)
        if val is None:
            return 0.0
        return val

    return {
        "aqi": aqi,
        "station": station,
        "city": city,
        "pm25": get_val("pm25"),
        "pm10": get_val("pm10"),
        "o3":   get_val("o3"),
        "co":   _get_iaqi_value(iaqi, "co") or 0.0,
        "no2":  _get_iaqi_value(iaqi, "no2") or 0.0,
        "so2":  _get_iaqi_value(iaqi, "so2") or 0.0,
        "temp":     _get_iaqi_value(iaqi, "t"),
        "humidity": _get_iaqi_value(iaqi, "h"),
        "wind":     _get_iaqi_value(iaqi, "w"),
    }