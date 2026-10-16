import json
import joblib
import numpy as np
import pandas as pd
import requests
from tensorflow import keras

from src.common import (RAW, FEATURES, WINDOW, LAT, LON, MODEL_PATH, SCALER_PATH,
                        CONFIG_PATH, clean, add_time_features)


def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    config = json.loads(CONFIG_PATH.read_text())
    return model, scaler, config


def predict_next_hour(df, model, scaler):
    df = add_time_features(clean(df))
    if len(df) < WINDOW:
        raise ValueError(f"Se necesitan al menos {WINDOW} filas horarias.")
    x = scaler.transform(df[FEATURES].tail(WINDOW))[None].astype("float32")
    y = float(model.predict(x, verbose=0)[0, 0])
    return y * scaler.scale_[0] + scaler.mean_[0]


def fetch_recent(days=3):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "hourly": ",".join(RAW),
        "past_days": days, "forecast_days": 1, "timezone": "America/Bogota",
    }, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    now = pd.Timestamp.now(tz="America/Bogota").tz_localize(None).floor("h")
    return df[df["time"] <= now].reset_index(drop=True)


if __name__ == "__main__":
    from src.common import load_data
    model, scaler, cfg = load_artifacts()
    df = load_data()
    print(f"Predicción próxima hora: {predict_next_hour(df.iloc[-WINDOW-1:-1], model, scaler):.2f} °C")
    print(f"Real: {df['temperature_2m'].iloc[-1]:.2f} °C")
