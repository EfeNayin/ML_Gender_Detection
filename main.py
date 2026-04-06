import cv2
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics

import preprocessing 


HAAR_CASCADE = './Classifier/haarcascade_frontalface_default.xml'

RAW_WOMEN_DIR = './data/women'
RAW_MEN_DIR = './data/men'

CROP_WOMEN_DIR = './crop_data/female'
CROP_MEN_DIR = './crop_data/male'

PCA_DATA_PATH = './data/data_pca_50_target.npz'
MODEL_SAVE_PATH = './model/model_svm.pickle'
PCA_MODEL_PATH = './model/pca_dict.pickle'


def extract_and_save_faces():
    """Ham fotoğraflardan yüzleri tespit edip keserek yeni klasörlere kaydeder."""
    preprocessing.create_directory(CROP_WOMEN_DIR)
    preprocessing.create_directory(CROP_MEN_DIR)

    w_path = preprocessing.get_image_paths(RAW_WOMEN_DIR)
    m_path = preprocessing.get_image_paths(RAW_MEN_DIR)

    print(f"Total women images found: {len(w_path)}")
    print(f"Total men images found: {len(m_path)}")

    def process_images(image_paths, save_folder, prefix):
        count = 0
        for i, img_path in enumerate(image_paths):
            try:
                face = preprocessing.face_detection(img_path, HAAR_CASCADE)
                if face is not None:
                    save_path = f"{save_folder}/{prefix}_{i}.jpg"
                    cv2.imwrite(save_path, face)
                    count += 1
                    if count % 300 == 0:
                        print(f"Successfully processed {count} {prefix} faces.")
            except Exception as e:
                print(f"Error reading ({img_path}): {e}")
        print(f"Finished. Total {count} {prefix} faces saved.")

    print("\n--- Processing Women's Photos ---")
    process_images(w_path, CROP_WOMEN_DIR, "female")

    print("\n--- Processing Men's Photos ---")
    process_images(m_path, CROP_MEN_DIR, "male")


def prepare_and_reduce_data():
    print("Loading cropped face data...")
    df = preprocessing.load_data_paths(CROP_WOMEN_DIR, CROP_MEN_DIR)
    X, y = preprocessing.create_training_data(df)

    print(f"Initial Training Data (X): {X.shape}")
    
    print("Starting PCA dimensionality reduction...")
    preprocessing.create_directory('./model')
    preprocessing.create_directory('./data')

    pca = PCA(n_components=50, whiten=True, svd_solver='auto')
    pca_data = pca.fit_transform(X)
    print(f"PCA completed. Reduced data shape: {pca_data.shape}")

    np.savez(PCA_DATA_PATH, pca_data, y)
    print(f"Transformed data saved as '{PCA_DATA_PATH}'.")

    pca_dict = {'pca': pca, 'mean_face': pca.mean_}
    with open(PCA_MODEL_PATH, 'wb') as f:
        pickle.dump(pca_dict, f)
    print("PCA Model saved successfully.")


def train_svm_model():
    print("Starting SVM Training Process...")
    
    data_pca = np.load(PCA_DATA_PATH)
    X = data_pca['arr_0'] 
    y = data_pca['arr_1'] 

    print(f"Training data loaded. Shape: {X.shape}")

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    print(f"Train set: {x_train.shape}, Test set: {x_test.shape}")

    model_svc = SVC(probability=True)
    param_grid = {
        'C': [0.5, 1, 10, 20, 30, 50],
        'kernel': ['rbf', 'poly'],
        'gamma': [0.1, 0.05, 0.01, 0.001, 0.002, 0.005],
        'coef0': [0, 1]
    }

    print("Running GridSearchCV for best hyperparameters...")
    model_grid = GridSearchCV(model_svc, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
    model_grid.fit(x_train, y_train)

    print(f"\nTraining Complete! Best parameters: {model_grid.best_params_}")
    
    model_final = model_grid.best_estimator_
    accuracy = model_final.score(x_test, y_test)
    print(f"Model Accuracy on Test Data: %{accuracy * 100:.2f}")

    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_final, f)
    print(f"Model saved successfully to '{MODEL_SAVE_PATH}'.")



if __name__ == "__main__":
    # extract_and_save_faces()

    # prepare_and_reduce_data()

    train_svm_model()