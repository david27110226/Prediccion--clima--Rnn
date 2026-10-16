# Predicción de temperatura en Bogotá con GRU

Proyecto de series temporales que predice la temperatura de la próxima hora usando las 48 horas anteriores y nueve variables meteorológicas/cíclicas.

## Dataset y experimento

- Fuente: Open-Meteo Historical API, Bogotá (4.61, -74.08), datos horarios del 2019-01-01 al 2025-12-31.
- 61.368 filas y 7 variables meteorológicas originales; no había valores nulos ni timestamps duplicados en la ejecución registrada.
- El notebook explora distribución, ciclos diarios/mensuales y correlaciones; ordena por fecha, elimina duplicados, interpola nulos y añade hora seno/coseno.
- División cronológica 70/15/15; `StandardScaler` se ajusta solo al bloque de entrenamiento. Se comparan GRU y LSTM con baseline de persistencia.
- Métricas de prueba registradas: baseline MAE 0.750 °C; GRU MAE 0.379 °C, RMSE 0.511 °C, R² 0.969; LSTM MAE 0.409 °C, RMSE 0.546 °C, R² 0.964. Se seleccionó GRU.

## Contenido

```text
notebook/experimentacion.ipynb  EDA, limpieza, entrenamiento y evaluación ejecutados
src/train.py                   Entrena y guarda modelo y reportes
src/prepare_artifacts.py        Reconstruye scaler/config con el CSV original
src/predict.py                 Carga artefactos y predice
app.py                          Aplicación Streamlit
data/bogota_hourly.csv          Dataset (descargar desde Colab o se descarga al entrenar)
models/                         Modelo Keras, scaler y configuración
reports/                        Curvas, evaluación y 10 predicciones de prueba
```

## Instalación y ejecución

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Si se tiene el ZIP descargado al final del notebook, extraer `data/`, `models/` y `reports/` en la raíz del proyecto. El CSV puede ubicarse también en `notebook/bogota_hourly.csv`; el código lo copiará a `data/`.

El modelo entrenado ya incluido es GRU. Si se recibió el modelo y el CSV, pero faltan el scaler/configuración, reconstruirlos con la división original:

```bash
python -m src.prepare_artifacts --data data/bogota_hourly.csv
```

Para entrenar de nuevo ambos modelos y regenerar todos los reportes:

```bash
python -m src.train --epochs 30
```

Predicción por consola y aplicación:

```bash
python -m src.predict
streamlit run app.py
```

La aplicación acepta una ventana del conjunto de prueba, un CSV propio (`time` y las siete columnas meteorológicas) o datos recientes de Open-Meteo. El CSV de entrada debe tener al menos 48 filas horarias.

## Resultados de prueba

Las diez predicciones están en `reports/predicciones_test.csv`; las métricas y gráficas están en `reports/`. Estos archivos se extrajeron de las salidas guardadas del notebook ejecutado. El notebook ahora también crea un ZIP con dataset, modelo, scaler, configuración y reportes para facilitar su transferencia desde Colab.
