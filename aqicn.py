from __future__ import annotations
import os
from typing import Any, Dict, Optional
import requests

AQICN_BASE = "https://api.waqi.info"

def fetch_aqicn(city: str = "Kuala Lumpur", token: Optional[str] = None) -> Dict[str, Any]:
    token = token or os.getenv("AQICN_TOKEN")
    url = f"{AQICN_BASE}/feed/{city}/"
    
    try:
        r = requests.get(url, params={"token": token}, timeout=10)
        raw = r.json()
        
        if raw.get("status") == "ok":
            d = raw.get("data") or {}
            forecast = d.get("forecast") or {}
            
            # Extract daily PM2.5 averages for history
            daily_data = forecast.get("daily", {}).get("pm25", [])
            
            # Map 'avg' values to the 'aqi_history' key your Flutter app expects
            aqi_history = [float(day['avg']) for day in daily_data if 'avg' in day][-7:]
            
            return {
                "aqi": d.get("aqi"),
                "aqi_history": aqi_history, # Matches your screenshot
                "station": d.get("city", {}).get("name"),
                "city": city
            }
        return {"aqi": None, "aqi_history": [], "station": "Unknown"}
    except Exception:
        return {"aqi": None, "aqi_history": [], "station": "Error"}