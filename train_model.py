import os
import pickle

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

import preprocessing

HAAR_CASCADE = "./model/haarcascade_frontalface_default.xml"

RAW_WOMEN_DIR = "./data/women"
RAW_MEN_DIR = "./data/men"

CROP_WOMEN_DIR = "./crop_data/female"
CROP_MEN_DIR = "./crop_data/male"

PCA_DATA_PATH = "./data/data_pca_150_target.npz"
MODEL_SAVE_PATH = "./model/model_svm.pickle"
PCA_MODEL_PATH = "./model/pca_dict.pickle"

N_COMPONENTS = 150
RANDOM_STATE = 42

def extract_and_save_faces():
    """Ham fotoğraflardan yüzleri tespit edip keserek yeni klasörlere kaydeder."""
    if cv2.CascadeClassifier(HAAR_CASCADE).empty():
        raise FileNotFoundError(
            f"Haar cascade okunamadı: {HAAR_CASCADE}\n"
            "Dosya yolunu kontrol edin."
        )

    preprocessing.create_directory(CROP_WOMEN_DIR)
    preprocessing.create_directory(CROP_MEN_DIR)


def extract_and_save_faces():
    """Detects and crops faces from raw photos, saving them to new folders."""
    preprocessing.create_directory(CROP_WOMEN_DIR)
    preprocessing.create_directory(CROP_MEN_DIR)

    w_path = preprocessing.get_image_paths(RAW_WOMEN_DIR)
    m_path = preprocessing.get_image_paths(RAW_MEN_DIR)

    print(f"Total women images found: {len(w_path)}")
    print(f"Total men images found: {len(m_path)}")

    def process_images(image_paths, save_folder, prefix):
        saved = 0
        for i, img_path in enumerate(image_paths):
            try:
                face = preprocessing.face_detection(img_path, HAAR_CASCADE)
            except Exception as exc:
                print(f"Error reading ({img_path}): {exc}")
                continue

            if face is None:
                continue

            save_path = os.path.join(save_folder, f"{prefix}_{i}.jpg")
            cv2.imwrite(save_path, face)
            saved += 1
            if saved % 300 == 0:
                print(f"  ...{saved} {prefix} faces saved")

        rate = saved / len(image_paths) * 100 if image_paths else 0
        print(f"Finished {prefix}: {saved}/{len(image_paths)} faces "
              f"detected ({rate:.1f}%)")
        return saved

    print("\n--- Processing Women's Photos ---")
    n_female = process_images(w_path, CROP_WOMEN_DIR, "female")

    print("\n--- Processing Men's Photos ---")
    n_male = process_images(m_path, CROP_MEN_DIR, "male")

    total = n_female + n_male
    if total:
        print(f"\nClass balance: {n_female / total * 100:.1f}% female / "
              f"{n_male / total * 100:.1f}% male")


def prepare_and_reduce_data():
    print("Loading cropped face data...")
    df = preprocessing.load_data_paths(CROP_WOMEN_DIR, CROP_MEN_DIR)
    X, y = preprocessing.create_training_data(df)

    print(f"Initial Training Data (X): {X.shape}")

    preprocessing.create_directory("./model")
    preprocessing.create_directory("./data")

    print(f"Starting PCA reduction to {N_COMPONENTS} components...")
    pca = PCA(n_components=N_COMPONENTS, whiten=True, svd_solver="auto",
              random_state=RANDOM_STATE)
    pca_data = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA completed. Reduced shape: {pca_data.shape}")
    print(f"Explained variance retained: {explained:.1f}%")

    np.savez(PCA_DATA_PATH, pca_data, y)
    print(f"Transformed data saved as '{PCA_DATA_PATH}'.")

    # mean_face is no longer saved: PCA.transform subtracts the mean itself,
    # subtracting it manually led to the mean being subtracted twice.
    with open(PCA_MODEL_PATH, "wb") as f:
        pickle.dump({"pca": pca, "n_components": N_COMPONENTS}, f)
    print("PCA Model saved successfully.")


def train_svm_model():
    print("Starting SVM Training Process...")

    data_pca = np.load(PCA_DATA_PATH)
    X = data_pca["arr_0"]
    y = data_pca["arr_1"]

    labels, counts = np.unique(y, return_counts=True)
    print(f"Training data loaded. Shape: {X.shape}")
    for label, count in zip(labels, counts):
        print(f"  {label}: {count} ({count / len(y) * 100:.1f}%)")

    majority = counts.max() / counts.sum() * 100
    print(f"Baseline (always predict majority class): {majority:.1f}%")
    print("The model should significantly outperform this baseline.\n")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train set: {x_train.shape}, Test set: {x_test.shape}")

    # class_weight='balanced' assigns more weight to the minority class (male),
    # preventing the model from systematically biasing towards the female class.
    model_svc = SVC(probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE)

    # rbf kernel does not use coef0 and degree; separate grids prevent redundant searching.
    param_grid = [
        {
            "kernel": ["rbf"],
            "C": [1, 10, 30, 50],
            "gamma": ["scale", 0.001, 0.005, 0.01],
        },
        {
            "kernel": ["poly"],
            "C": [1, 10],
            "degree": [2, 3],
            "gamma": ["scale", 0.005],
            "coef0": [0, 1],
        },
    ]

    print("Running GridSearchCV...")
    model_grid = GridSearchCV(
        model_svc,
        param_grid=param_grid,
        scoring="f1_macro",  # accuracy is misleading on imbalanced data
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    model_grid.fit(x_train, y_train)

    print(f"\nBest parameters: {model_grid.best_params_}")

    model_final = model_grid.best_estimator_
    y_pred = model_final.predict(x_test)

    accuracy = model_final.score(x_test, y_test)
    print(f"\nTest Accuracy: %{accuracy * 100:.2f}\n")

    print("Per-class performance:")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion matrix (row = true, column = predicted):")
    print(f"  labels: {list(model_final.classes_)}")
    print(confusion_matrix(y_test, y_pred, labels=model_final.classes_))

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model_final, f)
    print(f"\nModel saved successfully to '{MODEL_SAVE_PATH}'.")


if __name__ == "__main__":
    # Execute sequentially. Each step uses the output of the previous one.
    extract_and_save_faces()
    prepare_and_reduce_data()
    train_svm_model()