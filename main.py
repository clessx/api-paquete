from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
from PIL import Image
import numpy as np
import io, os, requests

app = FastAPI(title="POD-ML Paquetes API", version="v1.0")

# ==== CONFIGURACIÓN DEL MODELO ====
MODEL_PATH = "models/best_20251023-153753.keras"

# Si el modelo no existe, lo descarga automáticamente (ideal para Railway)
MODEL_URL = "https://drive.google.com/uc?export=download&id=TU_ID_DE_DRIVE"
os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("⬇️ Descargando modelo desde Google Drive...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Modelo descargado correctamente.")

print("🧠 Cargando modelo...")
model = keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado exitosamente.")
IMG_SIZE = (224, 224)

@app.get("/")
def home():
    return {"status": "API de clasificación de paquetes activa"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.expand_dims(np.array(image) / 255.0, axis=0)
    pred = model.predict(arr)[0][0]
    label = "valida" if pred > 0.5 else "no_valida"
    return {"prediction": label, "confidence": float(pred)}
