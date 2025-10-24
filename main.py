from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os

# === Inicialización de la API ===
app = FastAPI(title="API Clasificación Paquetes POD-ML", version="1.0")

# === Cargar modelo ===
MODEL_PATH = "models/paquete_classifier_v5_best_20251023-090007.keras"
model = load_model(MODEL_PATH)

# === Tamaño de imagen esperado ===
IMG_SIZE = (224, 224)

# === Endpoint raíz ===
@app.get("/")
async def root():
    return {"status": "API de clasificación activa", "model": "v5_best"}

# === Endpoint de predicción ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer bytes del archivo
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Redimensionar y normalizar
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar predicción
        prediction = model.predict(img_array)[0][0]
        clase = "valida" if prediction >= 0.5 else "no_valida"

        return JSONResponse({
            "prediction": clase,
            "confidence": float(prediction)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# === Ejecutar localmente ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
