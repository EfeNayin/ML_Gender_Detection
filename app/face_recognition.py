import os
import pickle

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

HAAR_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
SVM_PATH = os.path.join(MODEL_DIR, "model_svm.pickle")
PCA_PATH = os.path.join(MODEL_DIR, "pca_dict.pickle")

haar = cv2.CascadeClassifier(HAAR_PATH)

if haar.empty():
    raise RuntimeError(f"Haar cascade did not download: {HAAR_PATH}")

with open(SVM_PATH, "rb") as f:
    model_svm = pickle.load(f)

with open(PCA_PATH, "rb") as f:
    pca_models = pickle.load(f)

model_pca = pca_models["pca"]

COLORS = {"male": (255, 255, 0), "female": (255, 0, 255)}
FALLBACK_COLOR = (0, 255, 0)


def _preprocess(roi_gray):
    if roi_gray.shape[0] >= 100:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    roi_resize = cv2.resize(roi_gray, (100, 100), interpolation=interpolation)
    flat = roi_resize.flatten().reshape(1, -1).astype(np.float64) / 255.0
    return roi_resize, flat


def faceRecognitionPipeline(filename, path=True):
    if path:
        img = cv2.imread(filename)
    else:
        img = filename

    if img is None:
        raise ValueError(f"Image is not read: {filename}")

    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haar.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(40, 40),
    )

    predictions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_resize, flat = _preprocess(roi_gray)

        eigen_image = model_pca.transform(flat)
        reconstructed = model_pca.inverse_transform(eigen_image)

        result = model_svm.predict(eigen_image)[0]
        score = float(model_svm.predict_proba(eigen_image).max())

        eig_img = np.clip(reconstructed.reshape(100, 100) * 255.0, 0, 255).astype(
            np.uint8
        )

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