# One‑Page Summary: Iris MLOps System

**Goal:** Train and serve a classifier for the Iris dataset with full MLOps plumbing:
repo hygiene, experiment tracking (MLflow), containerized API, CI/CD, and monitoring.

## Architecture (high level)

- **Data layer**
  - `src/data.py` downloads Iris via `sklearn`, saves to `data/raw/iris.csv`.
  - Simple preprocessing (StandardScaler). Clean dir structure for raw/processed.
- **Model layer**
  - `src/train.py` trains **Logistic Regression** and **Random Forest** across small param grids.
  - Uses **MLflow** with a **SQLite backend** (`sqlite:///mlruns.db`) and a local artifact store (`./mlruns`).
  - Logs params/metrics and artifacts; selects best model by **F1-score** and **registers** it.
  - Persists **best model** + **scaler** to `models/` for the API.
- **Service layer**
  - **FastAPI** app (`app/main.py`) with **Pydantic** validation.
  - `/predict` (POST) accepts JSON; logs to **SQLite** and `logs/app.log`.
  - `/metrics` exposes Prometheus metrics; `/health` for readiness.
- **Packaging/Infra**
  - **Dockerfile** builds a slim image that serves with Uvicorn.
  - **GitHub Actions**: lint + test on every push; build & push Docker image to Docker Hub on `main` branch.
  - Scripts for local deploy: `scripts/deploy_local.sh`.
- **Monitoring**
  - Requests and predictions logged to SQLite (`predictions.db`) and `logs/app.log`.
  - Prometheus scrape-ready endpoint: `/metrics`.

## CI/CD flow

1. **Push PR/branch** → Lint + tests.
2. **Merge to main** → Build Docker → Push to Docker Hub (needs secrets).
