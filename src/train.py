from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .data import load_raw, preprocess
from .logger import logger

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(TRACKING_URI)
EXPERIMENT_NAME = "iris-classifiers"
mlflow.set_experiment(EXPERIMENT_NAME)


def get_param_grid():
    lr_grid = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs"],
        "max_iter": [200],
        "penalty": ["l2"],
    }
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 4],
        "random_state": [42],
    }
    return {
        "log_reg": (LogisticRegression, lr_grid),
        "rand_forest": (RandomForestClassifier, rf_grid),
    }


def iter_params(grid: dict):
    keys = sorted(grid.keys())
    from itertools import product

    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def train_and_log():
    df = load_raw()
    X_train, X_test, y_train, y_test, scaler, _ = preprocess(df)

    best = {"f1": -1, "run_id": None, "name": None, "model_obj": None}
    param_space = get_param_grid()

    input_example = pd.DataFrame(X_train[:2], columns=["f0", "f1", "f2", "f3"])

    for model_name, (ModelClass, grid) in param_space.items():
        for params in iter_params(grid):
            with mlflow.start_run(run_name=f"{model_name}") as run:
                model = ModelClass(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")

                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_macro", f1)

                dump(scaler, MODELS_DIR / "scaler.joblib")
                mlflow.log_artifact(
                    MODELS_DIR / "scaler.joblib", artifact_path="artifacts"
                )

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example,
                    registered_model_name=None,
                )

                if f1 > best["f1"]:
                    best.update(
                        {
                            "f1": f1,
                            "run_id": run.info.run_id,
                            "name": model_name,
                            "model_obj": model,
                        }
                    )

    dump(best["model_obj"], MODELS_DIR / "best_model.joblib")
    logger.info(
        f"Best model: {best['name']} with F1={best['f1']:.4f} (run_id={best['run_id']})"
    )

    try:
        model_uri = f"runs:/{best['run_id']}/model"
        result = mlflow.register_model(model_uri=model_uri, name="IrisClassifier")
        logger.info(
            f"Registered model 'IrisClassifier' (version={getattr(result, 'version', 'n/a')})"
        )
    except Exception as e:
        logger.warning(f"Model registry skipped/failed: {e}")

    return best


def main():
    train_and_log()
    logger.info("Training finished. See ./mlruns and ./models")


if __name__ == "__main__":
    main()
