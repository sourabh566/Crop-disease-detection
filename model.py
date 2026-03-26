import os
import numpy as np
import cv2
import tensorflow as tf
from config import DISEASE_DB, CLASS_NAMES, MODEL_INPUT_SHAPE, SAVED_MODEL_PATH


def load_disease_model():
    """
    Loads the trained PlantVillage model from SAVED_MODEL_PATH.
    If no trained model exists yet, falls back to the color-feature classifier.
    Run 'python train.py' once to generate the real model.
    """
    if os.path.exists(SAVED_MODEL_PATH):
        print(f"✅ Loading trained model from: {SAVED_MODEL_PATH}")
        model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        return ("nn", model)
    else:
        print("⚠️  No trained model found. Using color-feature classifier.")
        print("   Run 'python train.py' to train the model on PlantVillage data.")
        return ("color", None)


def _predict_with_nn(model, processed_image):
    """Run inference with the real trained neural network."""
    predictions = model.predict(processed_image, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])
    predicted_class = CLASS_NAMES[class_idx]
    return predicted_class, confidence


def _predict_with_color(processed_image):
    """
    Fallback: analyze color and texture features to estimate disease.
    Used only when the trained model is not available.
    """
    img = (processed_image[0] * 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(img_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    yellow_mask = cv2.inRange(img_hsv, np.array([20, 60, 100]), np.array([35, 255, 255]))
    brown_mask = cv2.inRange(img_hsv, np.array([0, 30, 20]), np.array([20, 200, 150]))
    dark_mask = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

    total = 224 * 224
    green_r = np.sum(green_mask > 0) / total
    yellow_r = np.sum(yellow_mask > 0) / total
    brown_r = np.sum(brown_mask > 0) / total
    dark_r = np.sum(dark_mask > 0) / total

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = float(np.var(gray)) / (255 * 255)

    classes = CLASS_NAMES
    scores = np.zeros(len(classes))

    # Simplified scoring for key diseases
    score_map = {
        "Apple___Apple_scab": dark_r * 2.0 + brown_r * 0.5 - green_r * 1.5,
        "Apple___Black_rot": brown_r * 2.5 + dark_r * 1.5 - green_r * 1.5,
        "Apple___Cedar_apple_rust": yellow_r * 2.0 + brown_r * 0.5 - green_r * 1.0,
        "Corn_(maize)___Common_rust_": yellow_r * 2.5 + texture * 0.5 - green_r * 0.5,
        "Corn_(maize)___Northern_Leaf_Blight": brown_r * 2.0 + texture * 1.0 - green_r * 1.0,
        "Grape___Black_rot": dark_r * 1.5 + brown_r * 1.5 - green_r * 1.2,
        "Potato___Early_blight": yellow_r * 1.5 + brown_r * 1.5 - green_r * 1.0,
        "Potato___Late_blight": brown_r * 2.0 + dark_r * 1.0 - yellow_r * 0.5,
        "Tomato___Bacterial_spot": dark_r * 2.0 - yellow_r * 0.5 + (1 - green_r) * 0.5,
        "Tomato___Late_blight": brown_r * 2.0 + dark_r * 1.0 - green_r * 0.5,
        "Squash___Powdery_mildew": (1 - green_r) * 1.5 + texture * 0.5,
        "Cherry_(including_sour)___Powdery_mildew": (1 - green_r) * 1.0 + texture * 0.5,
    }

    # Apply specific scores; default healthy score for all others is green-based
    for i, cls in enumerate(classes):
        if cls in score_map:
            scores[i] = score_map[cls]
        elif "healthy" in cls.lower():
            scores[i] = green_r * 3.0 - dark_r * 2.0 - brown_r * 2.0 - yellow_r * 1.5
        else:
            scores[i] = 0.05  # small base probability for unlisted diseases

    scores = scores - scores.max()
    exp_scores = np.exp(scores * 5)
    probabilities = exp_scores / exp_scores.sum()

    class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[class_idx])
    predicted_class = classes[class_idx]
    return predicted_class, confidence


def predict_disease(model_tuple, processed_image):
    """
    Dispatcher: routes to the real NN model or color-feature fallback.
    model_tuple is ("nn", model) or ("color", None).
    """
    mode, model = model_tuple
    if mode == "nn":
        return _predict_with_nn(model, processed_image)
    else:
        return _predict_with_color(processed_image)
