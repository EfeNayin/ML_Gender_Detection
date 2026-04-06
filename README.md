# ML Gender Detection Pipeline

An end-to-end machine learning pipeline for real-time gender classification from facial images. Built using classical computer vision techniques combined with dimensionality reduction and SVM classification.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-orange)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-green)

---

## 🚀 Key Features

- **Automated Face Detection** — Uses Haar Cascade classifier to detect and crop faces from raw images.
- **PCA Dimensionality Reduction** — Reduces 10,000-dimensional pixel data to 50 eigenfaces for efficient training.
- **SVM Classification** — Trains a Support Vector Machine with GridSearchCV hyperparameter tuning for optimal accuracy.
- **Modular Architecture** — Clean separation of preprocessing, training, and inference into dedicated modules.
- **Production-Ready Inference** — Real-time bounding box rendering with confidence scores on test images.

---

## 🛠 Tech Stack

| Component         | Tool                        |
|-------------------|-----------------------------|
| Language          | Python 3.10                 |
| Computer Vision   | OpenCV (cv2), Haar Cascade  |
| ML Framework      | Scikit-Learn (SVM, PCA)     |
| Data Processing   | NumPy, Pandas               |
| Visualization     | Matplotlib, Seaborn         |

---

## 📂 Project Structure

```
├── Classifier/
│   └── haarcascade_frontalface_default.xml   # Haar Cascade model
├── data/
│   ├── women/                                # Raw female images
│   └── men/                                  # Raw male images
├── crop_data/
│   ├── female/                               # Cropped female faces
│   └── male/                                 # Cropped male faces
├── model/
│   ├── model_svm.pickle                      # Trained SVM model
│   └── pca_dict.pickle                       # PCA model & mean face
├── test_images/                              # Input images for testing
├── test_results/                             # Output images with predictions
├── preprocessing.py                          # Face detection & data pipeline
├── main.py                                   # Training pipeline
├── test_image.py                             # Inference on single image
└── requirements.txt
```

---

## ⚙️ Pipeline Overview

The project runs in three sequential stages:

**1. Face Extraction**
Detects and crops faces from raw images using Haar Cascade, saving them into `crop_data/`.

**2. PCA Dimensionality Reduction**
Flattens each 100×100 grayscale face into a 10,000-dim vector, then reduces to 50 principal components using whitened PCA.

**3. SVM Training**
Trains an SVM classifier with `GridSearchCV` over C, kernel, gamma, and coef0 parameters to find the best configuration.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/EfeNayin/ML_Gender_Detection.git
cd ML_Gender_Detection

# Install dependencies
pip install -r requirements.txt
```

**Step 1 — Extract faces from raw data:**
```python
# In main.py, uncomment:
extract_and_save_faces()
```

**Step 2 — Run PCA reduction:**
```python
# In main.py, uncomment:
prepare_and_reduce_data()
```

**Step 3 — Train SVM model:**
```python
# In main.py, uncomment:
train_svm_model()
```

**Step 4 — Test on a single image:**
```bash
python test_image.py
```

---

## 📊 Model Configuration

| Parameter        | Search Space                          |
|------------------|---------------------------------------|
| `C`              | 0.5, 1, 10, 20, 30, 50               |
| `kernel`         | rbf, poly                             |
| `gamma`          | 0.1, 0.05, 0.01, 0.001, 0.002, 0.005 |
| `coef0`          | 0, 1                                  |
| `PCA components` | 50                                    |
| `Image size`     | 100 × 100 px (grayscale)              |
