from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

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
            # We use 'avg' for historical context, 'max' for conservative real-time fallback
            return _to_float(entry.get("avg"))
    return None

def fetch_aqicn(
    city: str = "Kuala Lumpur",
    token: Optional[str] = None,
    timeout: int = 12,
) -> Dict[str, Any]:
    token = token or os.getenv("AQICN_TOKEN")
    if not token:
        raise AQICNError("Missing AQICN_TOKEN environment variable.")

    url = f"{AQICN_BASE}/feed/{city}/"
    params = {"token": token}

    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        raise AQICNError(f"Network error calling AQICN: {e}")

    if raw.get("status") != "ok":
        raise AQICNError(f"AQICN error: {raw.get('data')}")

    d = raw.get("data") or {}
    iaqi = d.get("iaqi") or {}
    forecast = d.get("forecast") or {}

    # Extract current date for forecast fallback
    api_time_str = d.get("time", {}).get("s", "")
    current_date_str = api_time_str.split(" ")[0] if api_time_str else datetime.utcnow().strftime("%Y-%m-%d")

    def get_val(pol_key: str) -> float:
        val = _get_iaqi_value(iaqi, pol_key)
        if val is None:
            val = _get_forecast_value(forecast, pol_key, current_date_str)
        return float(val) if val is not None else 0.0

    # Environmental Multiplier Logic: 
    # Compares 7-day mean vs median to detect pollution volatility
    daily_pm25 = forecast.get("daily", {}).get("pm25", [])
    history_vals = [day['avg'] for day in daily_pm25 if 'avg' in day][-7:]
    
    multiplier = 1.0
    if len(history_vals) >= 3:
        import statistics
        mean_val = statistics.mean(history_vals)
        median_val = statistics.median(history_vals)
        # Higher multiplier if air quality is fluctuating significantly
        multiplier = round(mean_val / median_val if median_val > 0 else 1.0, 3)

    return {
        "aqi": _to_float(d.get("aqi")),
        "station": d.get("city", {}).get("name"),
        "pm25": get_val("pm25"),
        "pm10": get_val("pm10"),
        "temp": _get_iaqi_value(iaqi, "t"),
        "humidity": _get_iaqi_value(iaqi, "h"),
        "env_multiplier": multiplier  # Critical for /predict_env
    }