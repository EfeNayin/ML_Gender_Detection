import os
import pickle

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

HAAR_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
SVM_PATH = os.path.join(MODEL_DIR, "model_svm.pickle")
PCA_PATH = os.path.join(MODEL_DIR, "pca_dict.pickle")

# --- Detection parameters ----------------------------------------------------
# IMPORTANT: These values must be EXACTLY the same as those in preprocessing.py.
# If one changes, the other must change as well, otherwise the model must be retrained.
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (40, 40)
IMAGE_SIZE = (100, 100)

# --- Load models once -------------------------------------------------
haar = cv2.CascadeClassifier(HAAR_PATH)
if haar.empty():
    raise RuntimeError(f"Failed to load Haar cascade: {HAAR_PATH}")

with open(SVM_PATH, "rb") as f:
    model_svm = pickle.load(f)

with open(PCA_PATH, "rb") as f:
    pca_models = pickle.load(f)

model_pca = pca_models["pca"]

# NOTE: We DO NOT MANUALLY SUBTRACT the "mean_face" value. sklearn's PCA.transform()
# function subtracts the mean internally; subtracting it manually once more
# leads to subtracting the mean twice and completely incorrect predictions.

COLORS = {"male": (255, 255, 0), "female": (255, 0, 255)}  # BGR
FALLBACK_COLOR = (0, 255, 0)


def _prepare_face_vector(gray_face):
    """The exact same pipeline as preprocessing.prepare_face_vector."""
    equalized = cv2.equalizeHist(gray_face)

    if equalized.shape[0] >= IMAGE_SIZE[0]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    resized = cv2.resize(equalized, IMAGE_SIZE, interpolation=interpolation)
    vector = resized.flatten().astype(np.float32).reshape(1, -1) / 255.0
    return resized, vector


def faceRecognitionPipeline(filename, path=True):
    """Detects faces in the image, predicts gender, and draws bounding boxes/labels on it.

    Returns: (annotated image, list of predictions)
    """
    if path:
        buffer = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    else:
        img = filename

    if img is None:
        raise ValueError(f"Failed to read image: {filename}")

    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haar.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE,
    )

    predictions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_resize, vector = _prepare_face_vector(roi_gray)

        eigen_image = model_pca.transform(vector)
        reconstructed = model_pca.inverse_transform(eigen_image)

        result = model_svm.predict(eigen_image)[0]
        score = float(model_svm.predict_proba(eigen_image).max())

        eig_img = np.clip(
            reconstructed.reshape(IMAGE_SIZE) * 255.0, 0, 255
        ).astype(np.uint8)

        font_scale = max(0.45, w / 320.0)
        thickness = max(1, int(round(w / 200.0)))
        band_height = int(28 * font_scale) + 6
        color = COLORS.get(result, FALLBACK_COLOR)
        label = f"{result} {score * 100:.0f}%"

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness + 1)
        cv2.rectangle(img, (x, max(0, y - band_height)), (x + w, y), color, -1)
        cv2.putText(
            img,
            label,
            (x + 4, max(band_height - 8, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        predictions.append(
            {
                "roi": roi_resize,
                "eig_img": eig_img,
                "prediction_name": result,
                "score": score,
                "box": (int(x), int(y), int(w), int(h)),
            }
        )

    return img, predictions