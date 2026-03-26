import cv2
import numpy as np
from PIL import Image
from config import MODEL_INPUT_SHAPE

def preprocess_image(image: Image.Image):
    """
    Apply preprocessing techniques: resizing, normalization, and noise reduction.
    """
    # 1. Convert PIL to OpenCV format (BGR)
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 2. Resizing
    img_resized = cv2.resize(img_bgr, MODEL_INPUT_SHAPE)

    # 3. Noise Reduction (Gaussian Blur)
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # 4. Normalization (0-1 range)
    img_normalized = img_blurred.astype(np.float32) / 255.0

    # 5. Expand dimensions for batch [1, H, W, C]
    img_final = np.expand_dims(img_normalized, axis=0)
    
    return img_final

def augment_image(img_array):
    """
    Apply data augmentation using OpenCV and NumPy.
    Accepts and returns a uint8 BGR image array of shape (H, W, 3).
    Used during training to improve model robustness.
    Augmentations: Random horizontal/vertical flip, rotation (±20°), zoom (±20%).
    """
    import random

    # Random horizontal flip
    if random.random() > 0.5:
        img_array = cv2.flip(img_array, 1)

    # Random vertical flip
    if random.random() > 0.5:
        img_array = cv2.flip(img_array, 0)

    # Random rotation ±20 degrees
    angle = random.uniform(-20, 20)
    h, w = img_array.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_array = cv2.warpAffine(img_array, M, (w, h))

    # Random zoom ±20%
    zoom = random.uniform(0.8, 1.2)
    zh, zw = int(h * zoom), int(w * zoom)
    img_array = cv2.resize(img_array, (zw, zh))
    # Crop or pad back to original size
    if zoom > 1.0:
        y1 = (zh - h) // 2
        x1 = (zw - w) // 2
        img_array = img_array[y1:y1 + h, x1:x1 + w]
    else:
        pad_y = (h - zh) // 2
        pad_x = (w - zw) // 2
        img_array = cv2.copyMakeBorder(img_array, pad_y, h - zh - pad_y,
                                       pad_x, w - zw - pad_x, cv2.BORDER_REFLECT)

    return img_array

