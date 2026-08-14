"""Runs gender detection on a single image.

Usage:
    python test_image.py
    python test_image.py ./test_images/test_3.jpg
    python test_image.py photo.jpg --no-window

The processing pipeline is defined in app/face_recognition.py; this script
calls it. Thus, training, the web interface, and this test use the same steps.
"""

import argparse
import os
import sys

import cv2

from app.face_recognition import faceRecognitionPipeline

RESULT_FOLDER = "./test_results"
DEFAULT_IMAGE = "./test_images/test_6.jpg"


def run_test(image_path, show_window=True):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return 1

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    try:
        pred_img, predictions = faceRecognitionPipeline(image_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Detected faces: {len(predictions)}")

    if not predictions:
        print("No face detected. Try a front-facing photo where the face "
              "is clearly visible.")
        return 0

    for i, face in enumerate(predictions, start=1):
        print(f"  Face {i}: {face['prediction_name']} "
              f"(confidence: %{face['score'] * 100:.2f})")

    filename = os.path.basename(image_path)
    save_path = os.path.join(RESULT_FOLDER, f"result_{filename}")
    cv2.imwrite(save_path, pred_img)
    print(f"Output saved to: {save_path}")

    if show_window:
        cv2.imshow("Gender Detection Results", pred_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run gender detection on a single image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help=f"Path to the image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Save the result without opening a preview window.",
    )
    args = parser.parse_args()

    sys.exit(run_test(args.image, show_window=not args.no_window))