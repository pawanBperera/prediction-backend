# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import joblib, json
import numpy as np

from sklearn.pipeline import Pipeline

from fastapi.middleware.cors import CORSMiddleware

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH  = MODEL_DIR / "meta.json"

app = FastAPI(title="Prediction API", version="1.0")

# CORS: allow your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Request model (frontend-friendly keys)
class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=80)
    childish_diseases: str
    accident_or_serious_trauma: str
    high_fevers_last_year: str
    alcohol_consumption: str
    smoking_habit: str
    hours_sitting_per_day: float = Field(..., ge=0, le=24)

    class Config:
        extra = "forbid"

COL_MAP = {
    "age": "Age",
    "childish_diseases": "Childish diseases",
    "accident_or_serious_trauma": "Accident or serious trauma",
    "high_fevers_last_year": "High fevers in the last year",
    "alcohol_consumption": "Frequency of alcohol consumption",
    "smoking_habit": "Smoking habit",
    "hours_sitting_per_day": "Number of hours spent sitting per day",
}

clf: Pipeline | None = None
meta: Dict[str, Any] = {}


def load_artifacts() -> None:
    """(Re)load model + metadata if both files exist."""
    global clf, meta
    if MODEL_PATH.exists() and META_PATH.exists():
        clf = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    else:
        clf = None
        meta = {}


def get_model() -> Pipeline:
    """Return a non-None trained model or raise 400."""
    if clf is None:
        load_artifacts()
    if clf is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")
    return clf


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train_endpoint():
    # Import lazily to keep app import light
    from train import train as do_train
    metrics = do_train()
    load_artifacts()
    return {"status": "trained", "metrics": metrics}


@app.get("/model/info")
def model_info():
    load_artifacts()
    if not meta:
        raise HTTPException(status_code=404, detail="No model metadata found. Train first.")
    return meta


@app.post("/predict")
def predict(req: PredictRequest):
    model = get_model()

    row = {
        COL_MAP["age"]: req.age,
        COL_MAP["childish_diseases"]: req.childish_diseases.strip().lower(),
        COL_MAP["accident_or_serious_trauma"]: req.accident_or_serious_trauma.strip().lower(),
        COL_MAP["high_fevers_last_year"]: req.high_fevers_last_year.strip().lower(),
        COL_MAP["alcohol_consumption"]: req.alcohol_consumption.strip().lower(),
        COL_MAP["smoking_habit"]: req.smoking_habit.strip().lower(),
        COL_MAP["hours_sitting_per_day"]: req.hours_sitting_per_day,
    }
    x_df = pd.DataFrame([row])

    pred = model.predict(x_df)[0]
    proba_all = model.predict_proba(x_df)[0]
    classes = list(map(str, model.classes_))
    prob_map = {classes[i]: float(proba_all[i]) for i in range(len(classes))}
    prob_pred = prob_map[str(pred)]

    explanation = _top_contributors(model, x_df, top_k=3)

    return {
        "prediction": str(pred),
        "probability": round(prob_pred, 4),
        "probabilities": prob_map,
        "why": explanation,
    }


def _top_contributors(model: Pipeline, x_df: pd.DataFrame, top_k: int = 3) -> List[Dict[str, float]]:
    """
    Lightweight per-sample explanation for LogisticRegression pipelines.
    Falls back silently if unavailable.
    """
    try:
        pre = model.named_steps["pre"]
        lr = model.named_steps["model"]  # LogisticRegression expected
        x_tr = pre.transform(x_df)
        if hasattr(x_tr, "toarray"):
            x_tr = x_tr.toarray()
        coefs = lr.coef_[0]  # binary
        feats = pre.get_feature_names_out()
        contrib = coefs * x_tr[0]
        idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        out = [{"feature": str(feats[i]), "impact": float(round(contrib[i], 4))} for i in idx]
        return out
    except (AttributeError, ValueError, KeyError):

        return []
