import sqlite3
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

DB_PATH = Path("predictions.db")

def _connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    con = _connect()
    cur = con.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            predicted_label TEXT,
            probabilities TEXT
        );
        '''
    )
    con.commit()
    con.close()

def log_prediction(record: Dict[str, Any], predicted_label: str, probabilities_json: str):
    con = _connect()
    cur = con.cursor()
    cur.execute(
        '''
        INSERT INTO predictions (ts, sepal_length, sepal_width, petal_length, petal_width, predicted_label, probabilities)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        ''',
        (
            datetime.utcnow().isoformat(),
            record.get("sepal_length"),
            record.get("sepal_width"),
            record.get("petal_length"),
            record.get("petal_width"),
            predicted_label,
            probabilities_json,
        ),
    )
    con.commit()
    con.close()
