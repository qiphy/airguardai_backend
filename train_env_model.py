# backend/train_env_model.py
import sqlite3
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

DB_PATH = Path(__file__).parent / "airguard.db"
MODEL_PATH = Path(__file__).parent / "airguard_model.joblib"

LABELS = ["LOW", "MEDIUM", "HIGH"]

def label_from_pm25(pm25: float) -> int:
    # Still “real-data”: labels derived from measured PM2.5 values (rule-based)
    if pm25 < 35:
        return 0
    if pm25 < 75:
        return 1
    return 2

def fetch_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT pm25, pm10, o3, co, aqi
        FROM readings
        WHERE pm25 IS NOT NULL AND aqi IS NOT NULL
        ORDER BY ts ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

def main():
    rows = fetch_rows()
    if len(rows) < 200:
        raise SystemExit(f"Not enough real readings to train reliably. Have {len(rows)}; aim for 200+.")

    X = []
    y = []
    for pm25, pm10, o3, co, aqi in rows:
        X.append([pm25 or 0.0, pm10 or 0.0, o3 or 0.0, co or 0.0, aqi or 0.0])
        y.append(label_from_pm25(pm25))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
