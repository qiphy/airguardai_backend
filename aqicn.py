from __future__ import annotations

import os
from typing import Any, Dict, Optional
import requests

AQICN_BASE = "https://api.waqi.info"

class AQICNError(RuntimeError):
    pass

def _to_float(x: Any) -> Optional[float]:
    if x is None: return None
    try:
        return float(x)
    except Exception:
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
        raise AQICNError(f"Network error: {e}")

    if raw.get("status") != "ok":
        raise AQICNError(f"AQICN error: {raw.get('data')}")

    d = raw.get("data") or {}
    forecast = d.get("forecast") or {}

    # Extract 7-Day History from the daily forecast averages
    # We prioritize PM2.5 as it's the primary driver for AQI in Malaysia
    daily_pm25 = forecast.get("daily", {}).get("pm25", [])
    aqi_history = [float(day['avg']) for day in daily_pm25 if 'avg' in day][-7:]

    return {
        "aqi": _to_float(d.get("aqi")),
        "aqi_history": aqi_history,
        "station": d.get("city", {}).get("name"),
        "city": city
    }