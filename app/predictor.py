import os
import tensorflow as tf
import cv2
import numpy as np

# Resolve path relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "keras_model.h5")
LABELS_PATH = os.path.join(CURRENT_DIR, "..", "labels.txt")

# Load the trained model once globally
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

model = keras.models.load_model(MODEL_PATH)

def load_labels(labels_path):
    """Load labels dynamically from labels.txt."""
    if not os.path.exists(labels_path):
        return ["fresh", "rotten"]
    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].lower())
            else:
                labels.append(line.lower())
    return labels

labels = load_labels(LABELS_PATH)
try:
    rotten_idx = labels.index("rotten")
except ValueError:
    rotten_idx = 1

def predict(cropped_bgr_img):
    """
    Predict the probability score of rottenness for a cropped BGR image.
    Note: The model includes the preprocess_input layer inside it, so we pass raw float32 [0, 255] values.
    
    Parameters:
        cropped_bgr_img: NumPy BGR array from OpenCV.
        
    Returns:
        float: Sigmoid probability score representing the Rotten class (closer to 1.0 is Rotten).
    """
    try:
        # 1. Convert BGR to RGB
        rgb_img = cv2.cvtColor(cropped_bgr_img, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to 224x224 (required by MobileNetV2)
        resized_img = cv2.resize(rgb_img, (224, 224))
        
        # 3. Convert to float32 and normalize to [-1.0, 1.0] (Teachable Machine standard)
        img_array = resized_img.astype(np.float32)
        normalized_img_array = (img_array / 127.5) - 1.0
        
        # 4. Expand dimensions for batch size of 1 (1, 224, 224, 3)
        img_tensor = np.expand_dims(normalized_img_array, axis=0)
        
        # 5. Execute model prediction
        prediction = model.predict(img_tensor, verbose=0)
        
        # If output shape matches multiple classes, extract rotten score at rotten_idx
        if prediction.shape[-1] > rotten_idx:
            score = float(prediction[0][rotten_idx])
        else:
            score = float(prediction[0][0])
        return score
    except Exception as e:
        print(f"Prediction failure: {e}")
        return 0.5  # Return neutral score in case of error