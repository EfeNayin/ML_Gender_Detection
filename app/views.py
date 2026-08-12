import os
import uuid

import cv2
from flask import render_template, request
from werkzeug.utils import secure_filename

from app.face_recognition import faceRecognitionPipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "upload")
PREDICT_DIR = os.path.join(BASE_DIR, "static", "predict")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PREDICT_DIR, exist_ok=True)


def index():
    if request.method == "GET":
        return render_template("index.html")

    file = request.files.get("image_name")
    if file is None or file.filename.strip() == "":
        return render_template("index.html", error="No image selected.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template(
            "index.html",
            error=f"{ext} is not supported. Please upload JPG, PNG, BMP, or WEBP.",
        )

    _ensure_dirs()

    # To prevent files with the same name from overwriting each other 
    # and to avoid browser caching issues, we give each upload a unique prefix.
    stamp = uuid.uuid4().hex[:8]
    safe_name = secure_filename(file.filename)
    stored_name = f"{stamp}_{safe_name}"
    upload_path = os.path.join(UPLOAD_DIR, stored_name)
    file.save(upload_path)

    try:
        pred_img, predictions = faceRecognitionPipeline(upload_path)
    except Exception as exc:
        return render_template("index.html", error=f"Image could not be processed: {exc}")

    if not predictions:
        return render_template(
            "index.html",
            error="No face found in the image. Try an image where the face is clear and front-facing.",
        )

    result_name = f"pred_{stored_name}"
    cv2.imwrite(os.path.join(PREDICT_DIR, result_name), pred_img)

    report = []
    for i, face in enumerate(predictions):
        roi_name = f"roi_{i}_{stored_name}"
        eig_name = f"eig_{i}_{stored_name}"
        cv2.imwrite(os.path.join(PREDICT_DIR, roi_name), face["roi"])
        cv2.imwrite(os.path.join(PREDICT_DIR, eig_name), face["eig_img"])

        report.append(
            {
                "index": i + 1,
                "roi": roi_name,
                "eigen": eig_name,
                "name": face["prediction_name"],
                "score": round(face["score"] * 100, 1),
            }
        )

    return render_template("index.html", result=result_name, report=report)