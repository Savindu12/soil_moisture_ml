import os
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import firebase_admin
from firebase_admin import credentials, firestore

# ──────────────────────────────────────────────────────────────────────────────
# 1. Define Pydantic models for request validation (Pydantic v2)
# ──────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    series: conlist(item_type=float, min_length=30, max_length=30)  # exactly 30 floats

class AnomalyRequest(BaseModel):
    windows: conlist(
        item_type=conlist(item_type=float, min_length=30, max_length=30),
        min_length=1
    )  # list of at least 1 window of 30 floats

# ──────────────────────────────────────────────────────────────────────────────
# 2. Initialize FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Compute project root and paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIREBASE_JSON = os.path.join(
    BASE_DIR,
    "soil-temp-bc256-firebase-adminsdk-fbsvc-d1d59a5db9.json"
)
SCALE_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FORECASTER_PATH = os.path.join(BASE_DIR, "models", "forecaster.h5")
AUTOENC_PATH = os.path.join(BASE_DIR, "models", "autoencoder.h5")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Verify files exist
# ──────────────────────────────────────────────────────────────────────────────
for path in (FIREBASE_JSON, SCALE_PATH, FORECASTER_PATH, AUTOENC_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Initialize Firebase Admin
# ──────────────────────────────────────────────────────────────────────────────
cred = credentials.Certificate(FIREBASE_JSON)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Load scaler and models (inference only)
# ──────────────────────────────────────────────────────────────────────────────
scaler = joblib.load(SCALE_PATH)
forecaster = tf.keras.models.load_model(FORECASTER_PATH, compile=False)
autoencoder = tf.keras.models.load_model(AUTOENC_PATH, compile=False)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Optional root endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Soil Temperature API is up and running!"}

# ──────────────────────────────────────────────────────────────────────────────
# 8. Prediction endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(req: PredictRequest):
    arr = np.array(req.series, dtype=float).reshape(1, 30, 1)
    try:
        scaled_pred = float(forecaster.predict(arr)[0, 0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    temp_pred = float(scaler.inverse_transform([[scaled_pred]])[0, 0])
    return {"prediction_scaled": scaled_pred, "prediction": temp_pred}

# ──────────────────────────────────────────────────────────────────────────────
# 9. Anomaly detection endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/anomalies")
async def anomalies(req: AnomalyRequest):
    arr = np.array(req.windows, dtype=float).reshape(-1, 30, 1)
    try:
        recon = autoencoder.predict(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    errors = np.mean((recon - arr) ** 2, axis=(1, 2))
    threshold = float(np.percentile(errors, 95))
    flags = (errors > threshold).tolist()

    for idx, (err, is_anom) in enumerate(zip(errors, flags)):
        db.collection("anomalies").add({
            "index": idx,
            "error": float(err),
            "is_anomaly": bool(is_anom)
        })

    return {"threshold": threshold, "flags": flags}
