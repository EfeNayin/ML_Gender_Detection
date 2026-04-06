# ML Gender Detection 

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

A complete **Machine Learning pipeline** to detect human faces from raw images and classify them by gender using **Computer Vision** & **Statistical Learning**.

---

## 🚀 Features
- **Face Detection:** Haar Cascades for automated extraction.
- **Dimensionality Reduction:** PCA (Eigenfaces) for efficient handling of images.
- **Robust Classifier:** SVM with hyperparameter tuning via GridSearchCV.
- **Inference & Visualization:** Test scripts showing predictions with confidence scores.

---

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** OpenCV, Scikit-Learn, Pandas, NumPy
- **Techniques:** PCA, SVM, GridSearchCV, Haar Cascade Classifiers

---

## 📂 Project Structure

ML_Gender_Detection/
├─ main.py # Orchestrator: Extract -> Prepare -> Train
├─ preprocessing.py # Image processing & data loader
├─ test_image.py # Inference & visualization
├─ Classifier/ # Pre-trained Haar Cascade XML
├─ model/ # Pickled SVM & PCA models
├─ data/ # Raw images (Men/Women)
├─ test_results/ # Output images with predicted labels

