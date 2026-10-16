"""Recreate preprocessing artifacts for the supplied, already-trained model.

Run this when modelo.keras and bogota_hourly.csv were downloaded from Colab
but scaler.pkl/config.json were not. The scaler must be fitted on the original
training split used by the notebook.
"""
import argparse
import json

import joblib
from sklearn.preprocessing import StandardScaler

from src.common import (
    CONFIG_PATH, FEATURES, MODEL_PATH, RAW, ROOT, SCALER_PATH, WINDOW,
    add_time_features, clean, load_data, split_indices,
)


def main(csv_path=None):
    df = load_data(csv_path) if csv_path else load_data()
    df = add_time_features(clean(df))
    a, b = split_indices(len(df))
    scaler = StandardScaler().fit(df.loc[:a - 1, FEATURES])

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el modelo entrenado: {MODEL_PATH}")
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    # Results are those recorded in the executed notebook, on this dataset.
    results = {
        "baseline": {"MAE": 0.750, "RMSE": 1.005, "R2": 0.879},
        "gru": {"MAE": 0.379, "RMSE": 0.511, "R2": 0.969},
        "lstm": {"MAE": 0.409, "RMSE": 0.546, "R2": 0.964},
    }
    CONFIG_PATH.write_text(json.dumps({
        "window": WINDOW, "features": FEATURES, "best_model": "gru",
        "test_start": b, "n_rows": len(df), "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Artefactos listos: {SCALER_PATH} y {CONFIG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Ruta del CSV descargado desde Colab")
    main(parser.parse_args().data)
