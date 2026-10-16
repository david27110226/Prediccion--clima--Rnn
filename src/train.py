import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

from src.common import (ROOT, FEATURES, WINDOW, MODEL_PATH, SCALER_PATH, CONFIG_PATH,
                        load_data, clean, add_time_features, make_windows, split_indices)


def build_model(kind, window, n_feat):
    rnn = layers.GRU if kind == "gru" else layers.LSTM
    model = keras.Sequential([
        layers.Input((window, n_feat)),
        rnn(64, return_sequences=True),
        layers.Dropout(0.2),
        rnn(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name=kind)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }


def main(epochs=30, data=None):
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    keras.utils.set_random_seed(42)

    df = add_time_features(clean(load_data(data) if data else load_data()))
    n = len(df)
    a, b = split_indices(n)

    scaler = StandardScaler().fit(df.loc[:a - 1, FEATURES])
    arr = scaler.transform(df[FEATURES]).astype("float32")
    mu, sd = scaler.mean_[0], scaler.scale_[0]

    Xtr, ytr = make_windows(arr[:a])
    Xva, yva = make_windows(arr[a:b])
    Xte, yte = make_windows(arr[b:])
    inv = lambda v: v * sd + mu
    yte_c = inv(yte)

    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    results = {"baseline": metrics(yte_c, inv(Xte[:, -1, 0]))}
    histories, preds, models = {}, {}, {}

    for kind in ["gru", "lstm"]:
        model = build_model(kind, WINDOW, len(FEATURES))
        cb = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
        ]
        h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                      batch_size=128, callbacks=cb, verbose=2)
        p = inv(model.predict(Xte, verbose=0).ravel())
        results[kind] = metrics(yte_c, p)
        histories[kind], preds[kind], models[kind] = h.history, p, model
        print(kind, results[kind])

    best = min(["gru", "lstm"], key=lambda k: results[k]["MAE"])
    models[best].save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    CONFIG_PATH.write_text(json.dumps({
        "window": WINDOW, "features": FEATURES, "best_model": best,
        "test_start": int(b), "n_rows": int(n), "results": results,
    }, indent=2))
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for kind, h in histories.items():
        ax[0].plot(h["loss"], label=f"{kind} train")
        ax[0].plot(h["val_loss"], "--", label=f"{kind} val")
        ax[1].plot(np.array(h["mae"]) * sd, label=f"{kind} train")
        ax[1].plot(np.array(h["val_mae"]) * sd, "--", label=f"{kind} val")
    ax[0].set(title="Loss (MSE)", xlabel="Época")
    ax[1].set(title="MAE (°C)", xlabel="Época")
    ax[0].legend(); ax[1].legend()
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "training_curves.png", dpi=120); plt.close()

    p = preds[best]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(yte_c[:240], label="Real")
    ax[0].plot(p[:240], label=f"Predicho ({best.upper()})")
    ax[0].set(title="Primeras 240 horas de prueba", ylabel="°C"); ax[0].legend()
    ax[1].scatter(yte_c, p, s=2, alpha=0.3)
    lim = [yte_c.min(), yte_c.max()]
    ax[1].plot(lim, lim, "r--")
    ax[1].set(title="Real vs predicho", xlabel="Real °C", ylabel="Predicho °C")
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "pred_vs_real.png", dpi=120); plt.close()

    idx = np.linspace(0, len(yte_c) - 1, 10).astype(int)
    times = pd.to_datetime(df["time"]).iloc[b + WINDOW:].reset_index(drop=True)
    tabla = pd.DataFrame({
        "hora_objetivo": times.iloc[idx].values,
        "real_C": yte_c[idx].round(2),
        "predicho_C": p[idx].round(2),
    })
    tabla["error_C"] = (tabla.predicho_C - tabla.real_C).round(2)
    tabla.to_csv(ROOT / "reports" / "predicciones_test.csv", index=False)
    print(tabla)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    main(a.epochs, a.data)
