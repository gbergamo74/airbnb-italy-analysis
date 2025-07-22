import os, joblib, pandas as pd

_BASE = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_BASE, "ridge_pipeline.pkl")

def load_model():
    return joblib.load(_MODEL_PATH) if os.path.exists(_MODEL_PATH) else None

def predict_one(payload: dict) -> float:
    model = load_model()
    if model is None:
        raise RuntimeError("Model file ridge_pipeline.pkl non trovato")
    df = pd.DataFrame([payload])
    return float(model.predict(df)[0])
