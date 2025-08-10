# Iris MLOps Assignment (End‑to‑End)

This repo implements the complete assignment with: data/versioning, model training with MLflow,
API + Docker packaging, CI/CD via GitHub Actions, and logging/monitoring.

## Quick Start (local)

> Requires: Python 3.11+, Docker (for container steps), and Git.

```bash
# 1) Clone or unzip this repo, then:
cd iris-mlops-assignment

# 2) Create & activate a virtual env
python -m venv .venv
source ./.venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 3) Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4) Train + track experiments in MLflow (creates mlruns.db + mlruns/)
python src/train.py

# 5) (Optional) Launch MLflow UI to inspect runs & registered model
# Open http://127.0.0.1:5000 in a browser
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# 6) Run the API (FastAPI + Uvicorn)
uvicorn app.main:app --reload --port 8000

# 7) Try a prediction
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"records":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]}'

# 8) Prometheus metrics
curl http://127.0.0.1:8000/metrics
```

## Docker (local)

```bash
# Build
docker build -t yourdockerhubusername/iris-ml-api:latest .

# Run
docker run -p 8000:8000 yourdockerhubusername/iris-ml-api:latest

# Predict
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"records":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]}'

# Metrics
curl http://127.0.0.1:8000/metrics
```

## GitHub Actions CI/CD

- **CI**: On every push, workflow runs lint (flake8), format check (black --check), and tests (pytest).
- **CD**: On push to `main`, builds Docker image and pushes to Docker Hub as `latest` and with the commit SHA tag.
  Configure repo **Secrets**:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN` (a Docker Hub Access Token)

## Structure

```text
iris-mlops-assignment/
├── app/
│   └── main.py                # FastAPI service (with pydantic validation + Prometheus metrics)
├── data/
│   ├── processed/             # Saved processed data (if needed)
│   └── raw/                   # Saved raw data (iris.csv is produced on first train)
├── logs/
├── models/
│   ├── best_model.joblib      # Best model (written by train.py)
│   └── scaler.joblib          # Fitted StandardScaler
├── scripts/
│   ├── deploy_local.sh
│   └── run_api.sh
├── src/
│   ├── __init__.py
│   ├── data.py                # Load/save/preprocess iris dataset
│   ├── db.py                  # SQLite logging of predictions
│   ├── logger.py              # Loguru logger config
│   ├── schemas.py             # Pydantic request/response schemas
│   └── train.py               # MLflow experiments + model registration + persist best model
├── tests/
│   ├── test_api.py
│   └── test_train.py
├── .github/workflows/ci.yml   # Lint/Test + Docker build/push
├── Dockerfile
├── Makefile
├── requirements.txt
├── setup.cfg
├── summary.md                 # 1-page architecture summary
└── README.md
```

---

## How to deliver (what to submit)

- **GitHub repo**: Push this folder as-is (after you add your own remote).
- **Docker Hub**: After setting GH secrets, push to main to publish the image.
- **Summary**: `summary.md` is included (edit as needed).
- **5-min demo**: Use the outline at the end of `summary.md` for your screen recording.

### Optional: DVC (Data Versioning)

Because we use the small Iris dataset (from `sklearn`), DVC is **not required**. If you want
bonus points, you can init DVC and track `data/raw/iris.csv` like this:

```bash
dvc init
dvc add data/raw/iris.csv
git add data/raw/iris.csv.dvc .dvc .gitignore
git commit -m "Track raw iris data with DVC"
# Configure a remote (e.g., local dir or S3) then: dvc push
```
