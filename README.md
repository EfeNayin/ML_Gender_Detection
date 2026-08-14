# ML Gender Detection

A gender classifier for facial images, built with classical computer vision: Haar Cascade face detection, PCA (eigenfaces) for dimensionality reduction, and an SVM classifier. Includes a Flask web interface for uploading an image and inspecting the model's prediction.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-orange)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-green)
![Flask](https://img.shields.io/badge/Web-Flask-lightgrey)

**Test accuracy: 80.3%** against a 59.0% majority-class baseline. See [Results](#-results) and [Limitations](#-limitations) before using this for anything real.

---

## 🚀 Features

- **Automated face extraction** — Haar Cascade detection with largest-face selection, recovering ~83% of the raw dataset.
- **Histogram equalization** — normalizes lighting differences between studio shots and casual photos.
- **PCA dimensionality reduction** — compresses 10,000-dimensional pixel vectors into 150 eigenfaces, retaining 89.2% of variance.
- **Class-balanced SVM** — `class_weight='balanced'` prevents the classifier from collapsing onto the majority class.
- **Web interface** — upload an image and see the bounding box, the 100×100 grayscale crop the model actually receives, and the PCA reconstruction of that crop.

---

## 🛠 Tech Stack

| Component       | Tool                       |
|-----------------|----------------------------|
| Language        | Python 3.10                |
| Computer Vision | OpenCV (cv2), Haar Cascade |
| ML Framework    | Scikit-Learn (PCA, SVM)    |
| Data Processing | NumPy, Pandas              |
| Web Framework   | Flask, Jinja2              |

---

## 📂 Project Structure

```
├── app/
│   ├── face_recognition.py       # Inference pipeline
│   └── views.py                  # Upload handling and result rendering
├── templates/
│   ├── base.html
│   └── index.html
├── static/
│   ├── upload/                   # User-uploaded images
│   └── predict/                  # Annotated output
├── data/
│   ├── men/                      # Raw male images
│   ├── women/                    # Raw female images
│   └── data_pca_150_target.npz   # PCA-transformed training data
├── crop_data/
│   ├── female/                   # Cropped female faces
│   └── male/                     # Cropped male faces
├── model/
│   ├── haarcascade_frontalface_default.xml
│   ├── model_svm.pickle          # Trained SVM
│   └── pca_dict.pickle           # Fitted PCA
├── main.py                       # Flask entry point
├── preprocessing.py              # Shared image processing
├── train_model.py                # Training pipeline
├── test_image.py                 # CLI inference on a single image
└── requirements.txt
```

---

## ⚙️ How It Works

```
Haar Cascade  →  grayscale crop  →  equalizeHist  →  100×100  →  /255
              →  PCA (150 components)  →  SVM  →  label + probability
```

Training runs in three stages, all driven by `train_model.py`:

**1. Face extraction** — detects faces in the raw dataset and saves the cropped region. When multiple faces are present, the largest is selected. Detection rate: 86.1% for women, 80.6% for men.

**2. PCA reduction** — each cropped face is converted to grayscale, histogram-equalized, resized to 100×100, flattened into a 10,000-dim vector, and scaled to [0,1]. Whitened PCA then reduces this to 150 components.

**3. SVM training** — `GridSearchCV` searches separate parameter grids per kernel (an RBF kernel ignores `coef0` and `degree`, so searching them wastes fits). Scoring uses `f1_macro` rather than accuracy, since the classes are imbalanced.

### One rule matters more than any parameter

The detection and preprocessing chain must be **identical** in training and inference. Eigenface methods are highly sensitive to framing: change `scaleFactor` on one side only and faces land differently inside the 100×100 crop, which quietly destroys accuracy. `preprocessing.py` and `app/face_recognition.py` share the same constants for this reason. Changing one requires changing the other and retraining.

---

## 🚀 Getting Started

```bash
git clone https://github.com/EfeNayin/ML_Gender_Detection.git
cd ML_Gender_Detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Run the web app** (uses the pre-trained model in `model/`):

```bash
python main.py
```

Open `http://127.0.0.1:5000` and upload a front-facing photo.

**Run on a single image from the command line:**

```bash
python test_image.py ./test_images/test_6.jpg
```

Results are written to `test_results/`. Pass `--no-window` to skip the preview window.

**Retrain from scratch** — place images in `data/men/` and `data/women/`, then:

```bash
rm -rf crop_data          # Windows: Remove-Item -Recurse crop_data
python train_model.py
```

This runs all three stages in sequence. Face extraction takes a few minutes; the grid search takes 10–30 minutes depending on dataset size and core count.

---

## 📊 Results

Trained on 2,770 cropped faces (1,635 female / 1,135 male) extracted from 3,309 raw images. Evaluated on a stratified 20% holdout of 554 images.

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| female    | 0.832     | 0.835  | 0.834    | 327     |
| male      | 0.761     | 0.758  | 0.759    | 227     |
| **Macro avg** | **0.797** | **0.796** | **0.796** | **554** |

**Accuracy: 80.3%** — 21 points above the 59.0% majority-class baseline.

Confusion matrix:

|                   | predicted female | predicted male |
|-------------------|------------------|----------------|
| **actual female** | 273              | 54             |
| **actual male**   | 55               | 172            |

Best hyperparameters: `kernel='poly'`, `C=1`, `degree=3`, `gamma=0.005`, `coef0=1`.

Male recall (75.8%) trails female recall (83.5%) by roughly 8 points, reflecting the remaining 59/41 class imbalance in the training set.

---

## ⚠️ Limitations

**This is a learning project, not a deployable classifier.** At 80% accuracy, roughly one prediction in five is wrong.

**The eigenface approach has a ceiling, and this is close to it.** PCA on raw grayscale pixels captures whatever varies most across the dataset — and that is overwhelmingly lighting direction and head pose, not gender. The leading principal components encode illumination; gender-relevant structure (jawline, brow ridge, facial proportions) sits well below that in the variance ordering. Adding components or widening the hyperparameter search does not fix this, because the features themselves are measuring the wrong thing.

This is not a flaw in eigenfaces as such. The method worked well in its original 1991 setting: controlled lighting, consistent framing, aligned faces. Web-scraped photos with varied angles, lighting, and occlusion are outside what it was designed for.

**Other constraints:**

- Haar Cascade only detects roughly front-facing faces; profile views and heavy occlusion are missed entirely.
- Gender is treated as a binary label, which the classifier's design assumes but reality does not.
- Predictions reflect the demographics of the training data. Performance on groups underrepresented in that data will be worse, and the reported accuracy does not capture this.

**Getting substantially past this ceiling requires different features, not better tuning** — a small CNN or a pre-trained face embedding model would be the next step, and would plausibly reach 95%+.

---

## 📄 License

MIT