# src/train_models.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models

# ──────────────────────────────────────────────────────────────────────────────
# 1. Paths & directories
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load prepped data
# ──────────────────────────────────────────────────────────────────────────────
csv_path = os.path.join(DATA_DIR, "prepped.csv")
df = pd.read_csv(
    csv_path,
    index_col="datetime",
    parse_dates=["datetime"]
)

# Extract the scaled temperature series as shape (N, 1)
series = df["scaled"].values.reshape(-1, 1)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Create sliding-window sequences
# ──────────────────────────────────────────────────────────────────────────────
def create_sequences(data: np.ndarray, window: int = 30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

window_size = 30
X, y = create_sequences(series, window=window_size)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Train/test split
# ──────────────────────────────────────────────────────────────────────────────
split_idx  = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ──────────────────────────────────────────────────────────────────────────────
# 5. Model definitions
# ──────────────────────────────────────────────────────────────────────────────
def build_forecaster(window: int):
    inp = layers.Input(shape=(window, 1))
    x   = layers.LSTM(64, return_sequences=True)(inp)
    x   = layers.LSTM(32)(x)
    out = layers.Dense(1)(x)
    return models.Model(inputs=inp, outputs=out, name="lstm_forecaster")

def build_autoencoder(window: int):
    inp = layers.Input(shape=(window, 1))
    x   = layers.Conv1D(16, 3, padding="same", activation="relu")(inp)
    x   = layers.MaxPool1D(2, padding="same")(x)
    x   = layers.Conv1D(8, 3, padding="same", activation="relu")(x)
    x   = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(1, 3, padding="same")(x)
    return models.Model(inputs=inp, outputs=out, name="conv_autoencoder")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Build & compile
# ──────────────────────────────────────────────────────────────────────────────
forecaster  = build_forecaster(window_size)
forecaster.compile(optimizer="adam", loss="mse")

autoencoder = build_autoencoder(window_size)
autoencoder.compile(optimizer="adam", loss="mse")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Training
# ──────────────────────────────────────────────────────────────────────────────
print("Training LSTM forecaster...")
forecaster.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("Training Conv1D autoencoder...")
autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, X_test)
)

# ──────────────────────────────────────────────────────────────────────────────
# 8. Save models
# ──────────────────────────────────────────────────────────────────────────────
forecaster_path  = os.path.join(MODEL_DIR, "forecaster.h5")
autoencoder_path = os.path.join(MODEL_DIR, "autoencoder.h5")

forecaster.save(forecaster_path)
autoencoder.save(autoencoder_path)

print(f"Models saved to:\n • {forecaster_path}\n • {autoencoder_path}")
