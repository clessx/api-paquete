# -*- coding: utf-8 -*-
"""
TRAIN_POD_PACKAGES_V7.PY
Entrenamiento mejorado y documentado con MobileNetV2
Autor: Cristian Yáñez (CorreosChile)
"""

import os, csv, datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ============================================================
# 🔧 CONFIGURACIÓN GENERAL
# ============================================================
BASE_DIR = r"C:\Users\Gosu Station\Desktop\POD-ML-Paquetes\data\classified_v3"
OUTPUT_DIR = r"C:\Users\Gosu Station\Desktop\POD-ML-Paquetes\models"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results_history.csv")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "results", "graphs")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 20
EPOCHS_FINE = 40
LR_HEAD = 1e-4
LR_FINE = 1e-5
UNFREEZE_RATIO = 0.5  # 🔓 ahora desbloquea 50 % de capas
SEED = 123

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# ============================================================
# 🧠 CARGA Y DIVISIÓN DEL DATASET
# ============================================================
train_ds = keras.utils.image_dataset_from_directory(
    BASE_DIR, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = keras.utils.image_dataset_from_directory(
    BASE_DIR, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ============================================================
# 🎨 DATA AUGMENTATION
# ============================================================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.GaussianNoise(0.05)
], name="data_augmentation")

# ============================================================
# 🏗️ MODELO BASE
# ============================================================
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid",
                       kernel_regularizer=regularizers.l2(1e-4))(x)
model = keras.Model(inputs, outputs, name="paquete_classifier_v7")

# ============================================================
# ⚙️ COMPILACIÓN Y CALLBACKS
# ============================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
best_model_path = os.path.join(OUTPUT_DIR, f"best_{timestamp}.keras")

class BestEpochLogger(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        best_val = max(self.model.history.history.get("val_accuracy", [0]))
        best_epoch = self.model.history.history["val_accuracy"].index(best_val) + 1
        summary_file = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.txt")
        with open(summary_file, "w") as f:
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Val Accuracy: {best_val:.4f}\n")
        print(f"\n📄 Resumen guardado en: {summary_file}")

callbacks = [
    keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_accuracy",
                                    save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=3, verbose=1, min_lr=1e-6),
    keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, f"run_{timestamp}")),
    BestEpochLogger()
]

# ============================================================
# 🏋️ ENTRENAMIENTO FASE 1 — HEAD
# ============================================================
model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
              loss="binary_crossentropy", metrics=["accuracy"])
print("\n🚀 Entrenando capa superior...")
history1 = model.fit(train_ds, validation_data=val_ds,
                     epochs=EPOCHS_HEAD, callbacks=callbacks)

# ============================================================
# 🔓 FINE-TUNING FASE 2 — 50% CAPAS
# ============================================================
n_layers = len(base_model.layers)
unfreeze = int(n_layers * UNFREEZE_RATIO)
for layer in base_model.layers[:-unfreeze]:
    layer.trainable = False
base_model.trainable = True

# 🔧 Asegurar que BatchNorm no se congele completamente
for layer in base_model.layers[-unfreeze:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(optimizer=keras.optimizers.Adam(LR_FINE),
              loss="binary_crossentropy", metrics=["accuracy"])
print(f"\n🔓 Fine-tuning activado (últimas {unfreeze}/{n_layers} capas)...")
history2 = model.fit(train_ds, validation_data=val_ds,
                     epochs=EPOCHS_FINE, callbacks=callbacks)

# ============================================================
# 💾 GUARDADO FINAL Y MÉTRICAS
# ============================================================
final_path = os.path.join(OUTPUT_DIR, f"final_{timestamp}.keras")
model.save(final_path)

best_acc = max(history1.history['val_accuracy'] + history2.history['val_accuracy'])
final_acc = history2.history['val_accuracy'][-1]
best_epoch = (history1.history['val_accuracy'] + history2.history['val_accuracy']).index(best_acc) + 1

print(f"\n✅ Modelo final guardado: {final_path}")
print(f"🏆 Mejor modelo guardado en: {best_model_path}")
print(f"📈 Accuracy final: {final_acc:.4f} | Mejor accuracy: {best_acc:.4f} (época {best_epoch})")

# ============================================================
# 🗂️ REGISTRO HISTÓRICO
# ============================================================
row = [timestamp, round(best_acc,4), round(final_acc,4), best_epoch,
       len(train_ds)*BATCH_SIZE, len(val_ds)*BATCH_SIZE]
header = ["timestamp","best_val_acc","final_val_acc","best_epoch","train_size","val_size"]

if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(header)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f); writer.writerow(row)

print(f"\n🧾 Resultados agregados en {RESULTS_FILE}")

# ============================================================
# 📊 GENERACIÓN AUTOMÁTICA DE GRÁFICOS
# ============================================================
def plot_metric(history_list, metric, title, save_name):
    values = [val for hist in history_list for val in hist.history[metric]]
    val_values = [val for hist in history_list for val in hist.history[f"val_{metric}"]]
    epochs = range(1, len(values) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, values, 'b-', label=f'Train {metric}')
    plt.plot(epochs, val_values, 'r--', label=f'Val {metric}')
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    graph_path = os.path.join(GRAPH_DIR, f"{save_name}_{timestamp}.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"📊 {title} guardado en {graph_path}")

plot_metric([history1, history2], "accuracy", "Evolución Accuracy", "accuracy_plot")
plot_metric([history1, history2], "loss", "Evolución Loss", "loss_plot")

print("\n✅ Entrenamiento completado con éxito. Resultados y gráficos guardados.")
