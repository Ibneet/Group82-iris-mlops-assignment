from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from typing import List
from pathlib import Path
import json

import numpy as np
from joblib import load, dump
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.schemas import PredictRequest, PredictResponse
from src.logger import logger
from src.db import init_db, log_prediction

app = FastAPI(title="Iris ML API", version="1.0.0")

REQUEST_COUNTER = Counter("api_requests_total", "Total API requests", ["endpoint", "method"])
PRED_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")

MODELS_DIR = Path("models")
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

target_names = load_iris().target_names.tolist()

def ensure_model_on_startup():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    scaler = None
    if BEST_MODEL_PATH.exists() and SCALER_PATH.exists():
        try:
            model = load(BEST_MODEL_PATH)
            scaler = load(SCALER_PATH)
            logger.info("Loaded persisted best model + scaler from ./models")
        except Exception as e:
            logger.warning(f"Failed loading persisted model; training fallback. Error: {e}")
    if model is None or scaler is None:
        iris = load_iris()
        X = iris.data
        y = iris.target
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=200).fit(Xs, y)
        dump(model, BEST_MODEL_PATH)
        dump(scaler, SCALER_PATH)
        logger.info("Trained fallback LogisticRegression and saved to ./models")
    return model, scaler

model, scaler = ensure_model_on_startup()
init_db()

@app.get("/health")
def health():
    REQUEST_COUNTER.labels(endpoint="/health", method="GET").inc()
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
@PRED_LATENCY.time()
def predict(payload: PredictRequest):
    REQUEST_COUNTER.labels(endpoint="/predict", method="POST").inc()
    try:
        records = [r.model_dump() for r in payload.records]
        X = np.array([[r["sepal_length"], r["sepal_width"], r["petal_length"], r["petal_width"]] for r in records])
        Xs = scaler.transform(X)
        probs = model.predict_proba(Xs).tolist()
        preds = [target_names[int(max(range(len(p)), key=lambda i: p[i]))] for p in probs]
        for r, p, pr in zip(records, preds, probs):
            log_prediction(r, p, json.dumps(pr))
        return {"predictions": preds, "probabilities": probs}
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    REQUEST_COUNTER.labels(endpoint="/metrics", method="GET").inc()
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
