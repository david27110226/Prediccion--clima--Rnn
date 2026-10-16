import pandas as pd
import streamlit as st

from src.common import RAW, WINDOW, MODEL_PATH, SCALER_PATH, CONFIG_PATH, load_data
from src.predict import load_artifacts, predict_next_hour, fetch_recent

st.set_page_config(page_title="Predicción de temperatura", page_icon="🌡️")
st.title("🌡️ Predicción de temperatura (GRU/LSTM)")
st.caption(f"Predice la temperatura de la próxima hora a partir de las últimas {WINDOW} horas — Bogotá.")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


missing = [p.name for p in (MODEL_PATH, SCALER_PATH, CONFIG_PATH) if not p.exists()]
if missing:
    st.error("Faltan artefactos del modelo: " + ", ".join(missing))
    st.info(
        "Si ya tienes el CSV original, colócalo en data/bogota_hourly.csv "
        "y ejecuta `python -m src.prepare_artifacts`. "
        "El modelo .keras y el scaler deben corresponder al mismo experimento."
    )
    st.stop()

model, scaler, cfg = get_artifacts()
st.sidebar.write(f"Modelo: **{cfg['best_model'].upper()}**")
for k, v in cfg["results"][cfg["best_model"]].items():
    st.sidebar.write(f"{k}: {v:.3f}")
st.sidebar.write(f"MAE baseline: {cfg['results']['baseline']['MAE']:.3f}")

fuente = st.radio("Datos de entrada", ["Muestra del conjunto de prueba", "Subir CSV", "Datos recientes (Open-Meteo)"])

df_in, real = None, None

if fuente == "Muestra del conjunto de prueba":
    data = load_data()
    start = cfg["test_start"]
    n_opts = len(data) - start - WINDOW
    k = st.slider("Ventana de prueba", 0, n_opts - 1, 0)
    df_in = data.iloc[start + k: start + k + WINDOW]
    real = float(data["temperature_2m"].iloc[start + k + WINDOW])

elif fuente == "Subir CSV":
    st.write(f"Columnas requeridas: `time`, {', '.join(f'`{c}`' for c in RAW)} (mínimo {WINDOW} filas horarias).")
    f = st.file_uploader("CSV", type="csv")
    if f:
        df_in = pd.read_csv(f)

else:
    if st.button("Traer datos recientes"):
        try:
            st.session_state["recent"] = fetch_recent()
        except Exception as e:
            st.error(f"No se pudo consultar Open-Meteo: {e}")
    df_in = st.session_state.get("recent")

if df_in is not None:
    st.line_chart(df_in.set_index("time")["temperature_2m"].tail(WINDOW))
    try:
        pred = predict_next_hour(df_in, model, scaler)
        st.metric("Temperatura predicha (próxima hora)", f"{pred:.2f} °C",
                  delta=f"{pred - df_in['temperature_2m'].iloc[-1]:+.2f} °C vs última hora")
        if real is not None:
            st.metric("Valor real", f"{real:.2f} °C", delta=f"error {pred - real:+.2f} °C", delta_color="off")
    except Exception as e:
        st.error(str(e))
