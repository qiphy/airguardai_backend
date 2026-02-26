import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "airguard_model.joblib"

# Updated to include your new environmental factors
FEATURES = ["pm25", "pm10", "o3", "co", "aqi", "temperature", "humidity"]
LABELS = ["LOW", "MEDIUM", "HIGH"]

# Global variable to cache the model in memory
_MODEL_CACHE = None

def label_from_pm25(pm25: Any) -> str:
    """Standalone helper to get risk based strictly on PM2.5."""
    try:
        val = float(pm25)
        if val < 35:
            return "LOW"
        if val < 75:
            return "MEDIUM"
        return "HIGH"
    except (ValueError, TypeError):
        return "UNKNOWN"

def load_model():
    """Loads the model into memory once and caches it."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        return None
    
    try:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        logger.info("Environment model loaded successfully into memory.")
        return _MODEL_CACHE
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def predict_risk(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts health risk based on environmental features.
    Includes a heuristic fallback if the ML model is missing or fails.
    """
    model = load_model()
    
    # Prepare input array with default 0.0 for missing features
    # Note: Using .get() ensures we don't crash if temp/humidity are missing
    try:
        x_input = []
        for f in FEATURES:
            val = features.get(f)
            # Handle cases where value might be None or a string '-'
            try:
                x_input.append(float(val) if val is not None else 0.0)
            except (ValueError, TypeError):
                x_input.append(0.0)
        
        x = np.array([x_input], dtype=float)

        if model:
            probs = model.predict_proba(x)[0]
            idx = int(np.argmax(probs))
            
            return {
                "risk": LABELS[idx],
                "confidence": round(float(probs[idx]), 4),
                "probs": {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
                "method": "ml_inference"
            }
    except Exception as e:
        logger.warning(f"ML Inference failed, falling back to heuristic: {e}")

    # --- HEURISTIC FALLBACK ---
    # If the model isn't trained yet or fails, use the PM2.5 logic
    pm25 = features.get("pm25", 0.0)
    risk = "LOW"
    if pm25 >= 75: risk = "HIGH"
    elif pm25 >= 35: risk = "MEDIUM"

    return {
        "risk": risk,
        "confidence": 1.0,
        "method": "heuristic_fallback",
        "note": "ML model unavailable"
    }