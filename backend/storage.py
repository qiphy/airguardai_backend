import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "airguard.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS readings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT,
      location TEXT,
      aqi INTEGER,
      pm25 REAL,
      pm10 REAL,
      o3 REAL,
      co REAL,
      no2 REAL,
      so2 REAL,
      station TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT,
      name TEXT,
      profile TEXT,
      rating INTEGER,
      comment TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT,
      location TEXT,
      alert_type TEXT,
      message TEXT,
      severity TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
      key TEXT PRIMARY KEY,
      value INTEGER
    );
    """)

    conn.commit()
    conn.close()

def _inc_metric(key: str, n: int = 1):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO metrics(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=value+?",
                (key, n, n))
    conn.commit()
    conn.close()

def record_reading(row: dict):
    _inc_metric("readings_ingested", 1)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO readings(ts, location, aqi, pm25, pm10, o3, co, no2, so2, station)
      VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (
        row.get("ts"),
        row.get("location"),
        row.get("aqi"),
        row.get("pm25"),
        row.get("pm10"),
        row.get("o3"),
        row.get("co"),
        row.get("no2"),
        row.get("so2"),
        row.get("station"),
    ))
    conn.commit()
    conn.close()

def fetch_history(location: str, hours: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      SELECT ts, aqi, pm25, pm10, o3, co, no2, so2
      FROM readings
      WHERE location = ?
      ORDER BY ts DESC
      LIMIT ?
    """, (location, hours * 12))  # approx 12 points/hour if you refresh every 5 min
    rows = cur.fetchall()
    conn.close()
    rows.reverse()
    return [
        {"ts": r[0], "aqi": r[1], "pm25": r[2], "pm10": r[3], "o3": r[4], "co": r[5], "no2": r[6], "so2": r[7]}
        for r in rows
    ]

def add_feedback(name: str, profile: str, rating: int, comment: str):
    _inc_metric("feedback_count", 1)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO feedback(ts, name, profile, rating, comment)
      VALUES(?,?,?,?,?)
    """, (datetime.now(timezone.utc).isoformat(), name, profile, rating, comment))
    conn.commit()
    conn.close()

def add_alert(location: str, alert_type: str, message: str, severity: str):
    _inc_metric("alerts_fired", 1)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO alerts(ts, location, alert_type, message, severity)
      VALUES(?,?,?,?,?)
    """, (datetime.now(timezone.utc).isoformat(), location, alert_type, message, severity))
    conn.commit()
    conn.close()

def list_alerts(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      SELECT ts, location, alert_type, message, severity
      FROM alerts
      ORDER BY ts DESC
      LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [{"ts": r[0], "location": r[1], "type": r[2], "message": r[3], "severity": r[4]} for r in rows]

def get_metrics():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM metrics")
    rows = cur.fetchall()
    conn.close()
    return {k: v for (k, v) in rows}
