# backend/aqicn.py
#
# AQICN fetch helper for AirGuard AI
# - Fetches Kuala Lumpur (default) or a provided city keyword
# - Fallback 1: If "iaqi" (real-time) is missing, looks at "forecast" data.
# - Fallback 2: If data is still missing, defaults to 0.0 to prevent model crashes.

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
    """
    iaqi is usually like: {"pm25": {"v": 26}, ...}
    """
    obj = iaqi.get(key)
    if not isinstance(obj, dict):
        return None
    return _to_float(obj.get("v"))

def _get_forecast_value(forecast: Dict[str, Any], key: str, target_date: str) -> Optional[float]:
    """
    Fallback: Extracts 'max' value for the current date from the forecast block.
    """
    daily_data = forecast.get("daily", {}).get(key, [])
    if not isinstance(daily_data, list):
        return None
    
    # Look for the entry matching today's date
    for entry in daily_data:
        if entry.get("day") == target_date:
            return _to_float(entry.get("max"))
    
    return None

def fetch_aqicn(
    city: str = "Kuala Lumpur",
    token: Optional[str] = None,
    timeout: int = 12,
) -> Dict[str, Any]:
    """
    Fetch latest AQI/pollutant data.
    Ensures NO pollutants are None (NaN) by defaulting to 0.0.
    """
    token = token or os.getenv("AQICN_TOKEN")
    if not token:
        raise AQICNError("Missing AQICN_TOKEN environment variable.")

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
        raise AQICNError(f"AQICN status={status}, data={raw.get('data')}")

    d = raw.get("data") or {}
    iaqi = d.get("iaqi") or {}
    forecast = d.get("forecast") or {}

    # 1. Determine the Date for Forecast Fallback
    api_time_str = d.get("time", {}).get("s", "")
    current_date_str = None
    if api_time_str:
        try:
            current_date_str = api_time_str.split(" ")[0] 
        except Exception:
            pass
            
    if not current_date_str:
        current_date_str = datetime.utcnow().strftime("%Y-%m-%d")

    # 2. Extract Basic Info
    station = None
    city_obj = d.get("city")
    if isinstance(city_obj, dict):
        station = city_obj.get("name")

    aqi_val = d.get("aqi")
    aqi = _to_float(aqi_val)

    # 3. Helper: Realtime -> Forecast -> Default 0.0
    def get_val(pol_key: str) -> float:
        # Try real-time
        val = _get_iaqi_value(iaqi, pol_key)
        
        # Try forecast if real-time is missing
        if val is None and current_date_str:
            val = _get_forecast_value(forecast, pol_key, current_date_str)
            
        # Final fallback to 0.0 so the ML model doesn't crash with NaN
        if val is None:
            return 0.0
        return val

    out: Dict[str, Any] = {
        "aqi": aqi,
        "station": station,
        "city": city,

        # Pollutants (Forced to float, never None)
        "pm25": get_val("pm25"),
        "pm10": get_val("pm10"),
        "o3":   get_val("o3"),
        
        # These usually have no forecast, so we default directly to 0.0 if missing
        "co":   _get_iaqi_value(iaqi, "co") or 0.0,
        "no2":  _get_iaqi_value(iaqi, "no2") or 0.0,
        "so2":  _get_iaqi_value(iaqi, "so2") or 0.0,

        # Weather
        "temp":     _get_iaqi_value(iaqi, "t"),
        "humidity": _get_iaqi_value(iaqi, "h"),
        "wind":     _get_iaqi_value(iaqi, "w"),
    }

    return out