import numpy as np
import pandas as pd
import requests
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bogota_hourly.csv"
MODEL_PATH = ROOT / "models" / "modelo.keras"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
CONFIG_PATH = ROOT / "models" / "config.json"

RAW = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
]
FEATURES = RAW + ["hour_sin", "hour_cos"]
WINDOW = 48
LAT, LON = 4.61, -74.08


def download_data(path=DATA_PATH, start="2019-01-01", end="2025-12-31"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(RAW),
        "timezone": "America/Bogota",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data(path=DATA_PATH):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    # Colab downloads are often placed beside the notebook. Accept that
    # location and copy the data into the canonical project data directory.
    notebook_copy = ROOT / "notebook" / "bogota_hourly.csv"
    if notebook_copy.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(notebook_copy)
        df.to_csv(path, index=False)
        return df
    return download_data(path)


def clean(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    df = df[RAW].interpolate(limit_direction="both")
    return df.reset_index()


def add_time_features(df):
    df = df.copy()
    hour = pd.to_datetime(df["time"]).dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df


def make_windows(arr, window=WINDOW, target_idx=0):
    X = sliding_window_view(arr[:-1], (window, arr.shape[1]))[:, 0]
    y = arr[window:, target_idx]
    return X.astype("float32"), y.astype("float32")


def split_indices(n, train=0.70, val=0.15):
    a = int(n * train)
    b = int(n * (train + val))
    return a, b
