from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_and_save_iris(csv_path: Path = RAW_DIR / "iris.csv") -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    df.to_csv(csv_path, index=False)
    return df


def load_raw() -> pd.DataFrame:
    csv_path = RAW_DIR / "iris.csv"
    if not csv_path.exists():
        return load_and_save_iris(csv_path)
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return (X_train, X_test, y_train, y_test, scaler, iris_target_names())


def iris_target_names():
    iris = load_iris()
    return iris.target_names.tolist()
