"""
train.py — Fine-tune MobileNetV2 on the PlantVillage dataset.

DATASET SETUP:
    Download PlantVillage from Kaggle:
    https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

    After downloading, unzip and ensure 'color' subfolder is structured as:
        dataset/
            Apple___Apple_scab/
                img1.jpg ...
            Apple___Black_rot/
                ...
            Tomato___healthy/
                ...

    Then set DATASET_DIR below (default: 'dataset/color').

USAGE:
    python train.py

After training completes, restart the Streamlit app:
    streamlit run app.py
"""

import os
import tensorflow as tf
from config import MODEL_INPUT_SHAPE, SAVED_MODEL_PATH, CLASS_NAMES

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR     = "dataset"          # <-- folder with disease sub-folders
BATCH_SIZE      = 32
IMG_SIZE        = MODEL_INPUT_SHAPE  # (224, 224)
EPOCHS_FROZEN   = 5                  # Phase 1: head-only
EPOCHS_FINETUNE = 10                 # Phase 2: fine-tune top layers
VALIDATION_SPLIT = 0.2
NUM_CLASSES     = len(CLASS_NAMES)

# ─── Sanity check ─────────────────────────────────────────────────────────────
if not os.path.isdir(DATASET_DIR):
    print(f"\n❌  Dataset folder '{DATASET_DIR}' not found.")
    print("    Download from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("    Unzip it so that 'dataset/' contains one sub-folder per disease class.\n")
    raise SystemExit(1)

print(f"✅  Dataset folder found: {DATASET_DIR}")

# ─── Data Pipeline ────────────────────────────────────────────────────────────
print("📦  Building data pipelines...")
AUTOTUNE = tf.data.AUTOTUNE

ds_train = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Normalize to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
ds_val   = ds_val.map(lambda x, y: (normalization_layer(x), y),   num_parallel_calls=AUTOTUNE)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

ds_train = (
    ds_train
    .cache()
    .shuffle(1000)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
ds_val = ds_val.cache().prefetch(AUTOTUNE)

# ─── Model ────────────────────────────────────────────────────────────────────
print("🔧  Building MobileNetV2 model...")
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ─── Phase 1: Train head only ─────────────────────────────────────────────────
print(f"\n🚀  Phase 1: Training classification head ({EPOCHS_FROZEN} epochs)...")
model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_FROZEN)

# ─── Phase 2: Fine-tune top MobileNetV2 layers ───────────────────────────────
print(f"\n🔬  Phase 2: Fine-tuning top 30 layers ({EPOCHS_FINETUNE} epochs)...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
]

model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks,
)

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
model.save(SAVED_MODEL_PATH)
print(f"\n✅  Model saved → {SAVED_MODEL_PATH}")
print("   Restart Streamlit: streamlit run app.py")
