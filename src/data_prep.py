# data_prep.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Path to data folder
BASE = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(BASE, "data", "soil_temps_hourly.csv")

# 1. Load
df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")

# 2. (Optional) resample to daily if you want daily modeling:
# df = df.resample("D").mean()

# 3. Scale
scaler = MinMaxScaler()
df["scaled"] = scaler.fit_transform(df[["temperature"]])

# 4. Save prepped data & scaler
df.to_csv(os.path.join(BASE, "data", "prepped.csv"))
joblib.dump(scaler, os.path.join(BASE, "models", "scaler.pkl"))
