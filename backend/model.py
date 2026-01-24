import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).parent / "airguard_model.joblib"

FEATURES = ["pm25", "pm10", "o3", "co", "aqi"]

LABELS = ["LOW", "MEDIUM", "HIGH"]

def label_from_pm25(pm25: float) -> str:
    # Simple ground truth labeling (you can refine later)
    if pm25 < 35:
        return "LOW"
    if pm25 < 75:
        return "MEDIUM"
    return "HIGH"

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Env model missing. Run train_env_model.py after collecting real AQICN readings.")
    return joblib.load(MODEL_PATH)

def predict_risk(features: dict) -> dict:
    model = load_model()
    x = np.array([[features.get(f, 0.0) for f in FEATURES]], dtype=float)
    probs = model.predict_proba(x)[0]
    idx = int(np.argmax(probs))
    return {
        "risk": LABELS[idx],
        "confidence": float(probs[idx]),
        "probs": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    }
