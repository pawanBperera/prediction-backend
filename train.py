# train.py
import json
import time
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from pandas.api.types import is_bool_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/raw/fertility.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "meta.json"

TARGET = "Diagnosis"
POS_LABEL = "Altered"  # minority class

NUMERIC = ["Age", "Number of hours spent sitting per day"]
CATEGORICAL = [
    "Childish diseases",
    "Accident or serious trauma",
    "High fevers in the last year",
    "Frequency of alcohol consumption",
    "Smoking habit",
]


def train() -> Dict:
    df: pd.DataFrame = pd.read_csv(DATA_PATH).dropna(how="all")

    # Make target explicitly a Series, then force string labels
    target_series: pd.Series = df.loc[:, TARGET].squeeze()
    if is_bool_dtype(target_series):
        target = target_series.map({True: "True", False: "False"})
    else:
        target = target_series.astype("string")  # avoids PyCharm bool/astype warning

    features: pd.DataFrame = df.drop(columns=[TARGET])

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC),
            ("cat", cat_pipe, CATEGORICAL),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)

    # Metrics (binary, positive label = POS_LABEL)
    proba_pos = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(POS_LABEL)]
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        f"f1_{POS_LABEL}": round(float(f1_score(y_test, y_pred, pos_label=POS_LABEL)), 4),
        "roc_auc": round(
            float(
                roc_auc_score(pd.Series(y_test == POS_LABEL, dtype=int), proba_pos)
            ),
            4,
        ),
        "classes": list(map(str, pipeline.classes_)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "timestamp": int(time.time()),
    }

    # Persist model + metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    feature_names = pipeline.named_steps["pre"].get_feature_names_out().tolist()
    meta = {
        "target": TARGET,
        "positive_label": POS_LABEL,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "feature_names": feature_names,
        "metrics": metrics,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    m = train()
    print("Trained. Metrics:", m)
