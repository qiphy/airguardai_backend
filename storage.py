from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("DB_PATH", "airguard.db")


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _table_has_column(con: sqlite3.Connection, table: str, col: str) -> bool:
    cur = con.execute(f"PRAGMA table_info({table})")
    cols = [r["name"] for r in cur.fetchall()]
    return col in cols


def init_db() -> None:
    con = _conn()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                location TEXT NOT NULL,
                station TEXT,
                aqi REAL,
                pm25 REAL,
                pm10 REAL,
                o3 REAL,
                co REAL,
                no2 REAL,
                so2 REAL
            )
            """
        )

        # Add uid column if missing
        if not _table_has_column(con, "readings", "uid"):
            con.execute("ALTER TABLE readings ADD COLUMN uid INTEGER")

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tracked_locations (
                uid INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                created_ts TEXT NOT NULL
            )
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                name TEXT NOT NULL,
                profile TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT NOT NULL
            )
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """
        )

        con.commit()
    finally:
        con.close()


def record_reading(r: Dict[str, Any]) -> None:
    """
    Expected keys:
      ts, location, station, aqi, pm25, pm10, o3, co, no2, so2
      optional: uid (int)
    """
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO readings (ts, uid, location, station, aqi, pm25, pm10, o3, co, no2, so2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.get("ts"),
                r.get("uid"),
                r.get("location") or "Unknown",
                r.get("station"),
                r.get("aqi"),
                r.get("pm25"),
                r.get("pm10"),
                r.get("o3"),
                r.get("co"),
                r.get("no2"),
                r.get("so2"),
            ),
        )
        con.commit()
    finally:
        con.close()


def track_location(uid: int, name: str) -> None:
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO tracked_locations (uid, name, created_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET name=excluded.name
            """,
            (int(uid), name, datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
    finally:
        con.close()


def list_tracked_locations() -> List[Dict[str, Any]]:
    con = _conn()
    try:
        cur = con.execute(
            "SELECT uid, name, created_ts FROM tracked_locations ORDER BY created_ts DESC"
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def _since_iso(hours: int) -> str:
    since = datetime.now(timezone.utc) - timedelta(hours=int(hours))
    return since.isoformat()


def fetch_history(location: str, hours: int) -> List[Dict[str, Any]]:
    """
    Backward-compatible: history by location string.
    """
    con = _conn()
    try:
        cur = con.execute(
            """
            SELECT ts, uid, location, station, aqi, pm25, pm10, o3, co, no2, so2
            FROM readings
            WHERE location = ?
              AND ts >= ?
            ORDER BY ts ASC
            """,
            (location, _since_iso(hours)),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def fetch_history_by_uid(uid: int, hours: int) -> List[Dict[str, Any]]:
    con = _conn()
    try:
        cur = con.execute(
            """
            SELECT ts, uid, location, station, aqi, pm25, pm10, o3, co, no2, so2
            FROM readings
            WHERE uid = ?
              AND ts >= ?
            ORDER BY ts ASC
            """,
            (int(uid), _since_iso(hours)),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def add_feedback(name: str, profile: str, rating: int, comment: str) -> None:
    con = _conn()
    try:
        con.execute(
            """
            INSERT INTO feedback (ts, name, profile, rating, comment)
            VALUES (?, ?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), name, profile, int(rating), comment or ""),
        )
        con.commit()
    finally:
        con.close()


def list_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    con = _conn()
    try:
        cur = con.execute(
            """
            SELECT ts, level, message
            FROM alerts
            ORDER BY ts DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

def delete_history_by_uid(uid: int) -> None:
    con = _conn()
    try:
        con.execute("DELETE FROM readings WHERE uid = ?", (int(uid),))
        con.commit()
    finally:
        con.close()


def untrack_location(uid: int) -> None:
    con = _conn()
    try:
        con.execute("DELETE FROM tracked_locations WHERE uid = ?", (int(uid),))
        con.commit()
    finally:
        con.close()



def get_metrics() -> Dict[str, Any]:
    """
    Minimal metrics implementation to keep your /metrics endpoint working.
    """
    con = _conn()
    try:
        cur = con.execute("SELECT COUNT(*) AS c FROM readings")
        readings_count = int(cur.fetchone()["c"])

        cur = con.execute("SELECT COUNT(*) AS c FROM tracked_locations")
        tracked_count = int(cur.fetchone()["c"])

        return {
            "readings_count": readings_count,
            "tracked_locations_count": tracked_count,
            "db_path": DB_PATH,
        }
    finally:
        con.close()
