import numpy as np
import pickle
import cv2
import os


image_path = './test_images/test_6.jpg' 

result_folder = './test_results'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    print(f"Results will be saved to '{result_folder}'.")

haar = cv2.CascadeClassifier('./Classifier/haarcascade_frontalface_default.xml')

# SVM 
model_svm = pickle.load(open('./model/model_svm.pickle', mode='rb'))

# PCA 
pca_models = pickle.load(open('./model/pca_dict.pickle', mode='rb'))
model_pca = pca_models['pca']
mean_face_arr = pca_models['mean_face']


img = cv2.imread(image_path)

if img is None:
    print(f"Error! İmage was not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = haar.detectMultiScale(gray, 1.5, 3)
print(f"{len(faces)} faces detected.")

#BGR
colors = {
    'male': (255, 255, 0),    
    'female': (255, 0, 255)   
}

for x, y, w, h in faces:

    roi = gray[y:y+h, x:x+w]

    roi = roi / 255.0
    
    if roi.shape[1] > 100:
        roi_resize = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        roi_resize = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_CUBIC)
        
    roi_reshape = roi_resize.reshape(1, 10000)

    eigen_image = model_pca.transform(roi_reshape)
   
    results = model_svm.predict(eigen_image)
    prob_score = model_svm.predict_proba(eigen_image)
    prob_score_max = prob_score.max()
    
    prediction_name = results[0]
    
    print(f"Result: {prediction_name} (Accuracy: %{prob_score_max*100:.2f})")
    
    color = colors.get(prediction_name, (0, 255, 0))
    text = f"{prediction_name} : %{prob_score_max*100:.0f}"
    
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    
    cv2.rectangle(img, (x, y-40), (x+w, y), color, -1)

    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

filename = os.path.basename(image_path)
save_path = os.path.join(result_folder, f"result_{filename}")

cv2.imwrite(save_path, img)
print(f"Result saved to: {save_path}")

cv2.imshow('Test Results', img)
cv2.waitKey(0)
cv2.destroyAllWindows()