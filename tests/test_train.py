from pathlib import Path
from src.train import main

def test_training_produces_artifacts():
    main()
    assert Path("models/best_model.joblib").exists()
    assert Path("models/scaler.joblib").exists()
