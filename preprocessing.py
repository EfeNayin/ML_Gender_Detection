import os

import cv2
import numpy as np
import pandas as pd

# --- Detection parameters ----------------------------------------------------
# IMPORTANT: These values must be EXACTLY the same as those in app/face_recognition.py.
# If training crops and prediction crops are generated with different parameters,
# the faces will fit differently inside the 100x100 square and the model will degrade.
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (40, 40)

IMAGE_SIZE = (100, 100)
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def get_image_paths(directory_path):
    """Collects all images in the directory (not just .jpg)."""
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(
            os.path.join(directory_path, name)
            for name in os.listdir(directory_path)
            if name.lower().endswith(pattern[1:])
        )
    return sorted(set(paths))


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")


def _read_image(path):
    """Safe image reading that also works with paths containing Turkish/non-ASCII characters."""
    try:
        buffer = np.fromfile(path, np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _largest_face(faces):
    """If there are multiple faces, select the largest one (eliminates background faces)."""
    return max(faces, key=lambda box: box[2] * box[3])


def face_detection(image_path, haar_path):
    img = _read_image(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {haar_path}")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE,
    )

    if len(faces) == 0:
        return None

    x, y, w, h = _largest_face(faces)
    return img[y : y + h, x : x + w]


def prepare_face_vector(gray_face):
    """Converts the grayscale face crop into a 1x10000 vector to be fed into the model.

    This function is a COMMON step for both training and prediction. If the pipeline
    changes, it must be updated here so the two sides do not diverge.
    """
    # Histogram equalization: neutralizes lighting differences. It brings a photo
    # taken under studio lighting and a dim selfie to the same scale.
    equalized = cv2.equalizeHist(gray_face)

    if equalized.shape[0] >= IMAGE_SIZE[0]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    resized = cv2.resize(equalized, IMAGE_SIZE, interpolation=interpolation)
    return resized, resized.flatten().astype(np.float32) / 255.0


def process_image_for_training(path):
    img = _read_image(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, vector = prepare_face_vector(gray)
    return vector


def load_data_paths(female_folder, male_folder):
    female_files = get_image_paths(female_folder)
    male_files = get_image_paths(male_folder)

    df_female = pd.DataFrame(female_files, columns=["filepath"])
    df_female["gender"] = "female"

    df_male = pd.DataFrame(male_files, columns=["filepath"])
    df_male["gender"] = "male"

    df = pd.concat([df_female, df_male], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset loaded: {len(df)} images "
          f"({len(df_female)} female / {len(df_male)} male)")
    return df


def create_training_data(df):
    data_list = []
    labels_list = []
    skipped = 0

    print("Processing images for training, please wait...")

    for _, row in df.iterrows():
        vector = process_image_for_training(row["filepath"])
        if vector is None:
            skipped += 1
            continue
        data_list.append(vector)
        labels_list.append(row["gender"])

    X = np.array(data_list, dtype=np.float32)
    y = np.array(labels_list)

    if skipped:
        print(f"Warning: {skipped} images could not be read and were skipped.")

    print(f"Final Data Shape: {X.shape}")
    return X, y