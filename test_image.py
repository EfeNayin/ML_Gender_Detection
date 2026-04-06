import numpy as np
import pickle
import cv2
import os

IMAGE_PATH = './test_images/test_6.jpg' 
RESULT_FOLDER = './test_results'
HAAR_CASCADE_PATH = './Classifier/haarcascade_frontalface_default.xml'
MODEL_PATH = './model/model_svm.pickle'
PCA_DICT_PATH = './model/pca_dict.pickle'

COLORS = {
    'male': (255, 255, 0),    
    'female': (255, 0, 255)  
}

def run_test():
    
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
        print(f"Directory created: '{RESULT_FOLDER}'")

    try:
        haar = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        model_svm = pickle.load(open(MODEL_PATH, mode='rb'))
        pca_models = pickle.load(open(PCA_DICT_PATH, mode='rb'))
        model_pca = pca_models['pca']
    except Exception as e:
        print(f"Error loading models or classifier: {e}")
        return

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    print(f"Detected faces: {len(faces)}")

    for x, y, w, h in faces:
        roi = gray[y:y+h, x:x+w] / 255.0
        
        interp = cv2.INTER_AREA if roi.shape[1] > 100 else cv2.INTER_CUBIC
        roi_resize = cv2.resize(roi, (100, 100), interpolation=interp)
        
        roi_reshape = roi_resize.reshape(1, 10000)
        eigen_image = model_pca.transform(roi_reshape)
        
        prediction = model_svm.predict(eigen_image)[0]
        prob_score = model_svm.predict_proba(eigen_image).max()
        
        print(f"Prediction: {prediction} (Confidence: %{prob_score*100:.2f})")
        
        color = COLORS.get(prediction, (0, 255, 0))
        label_text = f"{prediction.capitalize()}: {prob_score*100:.0f}%"
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color, -1)
        cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    filename = os.path.basename(IMAGE_PATH)
    save_path = os.path.join(RESULT_FOLDER, f"result_{filename}")
    cv2.imwrite(save_path, img)
    print(f"Output saved to: {save_path}")

    cv2.imshow('Gender Detection Results', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()