# -*- coding: utf-8 -*-
"""
API Paquete - Clasificación de imágenes válidas / no válidas
Modelo: paquete_classifier_v5_best_20251023-090007.keras
Autor: Cristian Yáñez (CorreosChile)
"""

import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import uvicorn

# ============================================================
# ⚙️ CONFIGURACIÓN
# ============================================================
MODEL_PATH = r"C:\Users\Gosu Station\Desktop\POD-ML-Paquetes\models\paquete_classifier_v5_best_20251023-090007.keras"
IMG_SIZE = (224, 224)

# ============================================================
# 🧠 CARGA DEL MODELO
# ============================================================
print(f"📦 Cargando modelo desde: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente y listo para servir.")

# ============================================================
# 🚀 API FASTAPI
# ============================================================
app = FastAPI(
    title="API Clasificación de Paquetes - CorreosChile",
    description="Servicio que clasifica imágenes como 'válida' o 'no válida' usando modelo v5_best",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "API de Clasificación POD funcionando correctamente."}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Recibe una imagen y retorna su clasificación ('valida' o 'no_valida')
    junto con la probabilidad estimada.
    """
    try:
        # Leer bytes de la imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)

        # Convertir a array y aplicar el preprocesamiento usado en el entrenamiento
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = preprocess_input(img_array)  # 👈 IMPORTANTE

        # Realizar la predicción
        prediction = model.predict(img_array)[0][0]
        label = "valida" if prediction > 0.5 else "no_valida"

        return JSONResponse({
            "prediction": label,
            "confidence": float(prediction)
        })

    except Exception as e:
        return JSONResponse(
            content={"error": f"Error procesando la imagen: {str(e)}"},
            status_code=500
        )


# ============================================================
# 🧩 EJECUCIÓN LOCAL
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
