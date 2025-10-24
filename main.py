# -*- coding: utf-8 -*-
"""
main.py — API de Clasificación de Paquetes (POD-ML)
Versión 2 — Corrige el problema de predicciones "todo no_valida"
Autor: Cristian Yáñez
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
from io import BytesIO

# === CONFIGURACIÓN ===
MODEL_PATH = "models/paquete_classifier_v5_best_20251023-090007.keras"
IMG_SIZE = (224, 224)
THRESHOLD = 0.4  # 🔧 umbral ajustado según análisis local

# === INICIALIZACIÓN DE API ===
app = FastAPI(title="POD-ML Paquete Classifier API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes limitarlo a tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CARGA DEL MODELO ===
print(f"📦 Cargando modelo desde: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente.")


# === ENDPOINT PRINCIPAL ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve la predicción de si es válida o no válida.
    """

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img)

        # ✅ Normalización EXACTA del entrenamiento
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # === PREDICCIÓN ===
        pred = model.predict(img_array, verbose=0)[0][0]
        clase = "valida" if pred > THRESHOLD else "no_valida"

        return {
            "prediction": clase,
            "confidence": float(pred)
        }

    except Exception as e:
        return {"error": f"❌ Error procesando la imagen: {str(e)}"}


@app.get("/")
async def root():
    return {
        "message": "POD-ML Paquete Classifier API v2 está corriendo correctamente 🚀",
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD
    }
