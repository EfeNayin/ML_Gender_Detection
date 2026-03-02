import cv2
import preprocessing
import os
import pickle
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics


haarcascade = r'.\Classifier\haarcascade_frontalface_default.xml'
w_data = r'.\data\women'
m_data = r'.\data\men'


save_women = './crop_data/female'
save_men = './crop_data/male'


preprocessing.create_directory(save_women)
preprocessing.create_directory(save_men)

w_path = preprocessing.get_image_paths(w_data)
m_path = preprocessing.get_image_paths(m_data)

print(f"The number of images in women folder: {len(w_path)}")
print(f"The number of images in men folder: {len(m_path)}")

def process_and_save(image_paths, save_folder, prefix, haar_path):
    count = 0
    for i, img_path in enumerate(image_paths):
        try:
            face = preprocessing.face_detection(img_path, haar_path)
            
            if face is not None:
                save_path = f"{save_folder}/{prefix}_{i}.jpg"
                
                cv2.imwrite(save_path, face)
                count += 1
                if count % 300 == 0:
                    print(f"Successfully processed {count} {prefix} images.")
        except Exception as e:
            print(f"Error ({img_path}): {e}")

    print(f"Total {count} {prefix} faces saved successfully.")

print("--- Women's Photos are Processed ---")
#process_and_save(w_path, save_women, "female",haarcascade)

print("\n--- Men's Photos are Processed ---")
#process_and_save(m_path,save_men, "male", haarcascade)


#print("Loading data...")
#df = preprocessing.load_data_paths(save_women, save_men)

#X, y = preprocessing.create_training_data(df)


#print(f"Training Data (X): {X.shape}")
#print(f"Label(y): {y.shape}")


# --- ADIM 2: PCA (BOYUT İNDİRGEME) ---
#print("Starting PCA...")

#preprocessing.create_directory('./model')
#preprocessing.create_directory('./data')


#pca = PCA(n_components=50, whiten=True, svd_solver='auto')

#pca_data = pca.fit_transform(X)

#print(f"PCA completed. New data shape: {pca_data.shape}")

# --- ADIM 3: KAYIT İŞLEMLERİ ---

#np.savez('./data/data_pca_50_target', pca_data, y)
#print("Transformed data saved as './data/data_pca_50_target.npz'.")


#pca_dict = {'pca': pca, 'mean_face': pca.mean_}

#with open('./model/pca_dict.pickle', 'wb') as f:
    #pickle.dump(pca_dict, f)

#print("PCA Model saved as './model/pca_dict.pickle'.")


# SVM MODEL TRAINING

print("Starting SVM Training...")


data_pca = np.load('./data/data_pca_50_target.npz')
X = data_pca['arr_0'] 
y = data_pca['arr_1'] 

print(f"Training data loaded.Shape: {X.shape}")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(f"Train size: {x_train.shape}, Test size: {x_test.shape}")

model_svc = SVC(probability=True)

param_grid = {
    'C': [0.5, 1, 10, 20, 30, 50],
    'kernel': ['rbf', 'poly'],
    'gamma': [0.1, 0.05, 0.01, 0.001, 0.002, 0.005],
    'coef0': [0, 1]
}

print("Searching for best hyperparameters via GridSearchCV...")
model_grid = GridSearchCV(model_svc, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

model_grid.fit(x_train, y_train)

print("Finish!")
print(f"Best parameters found: {model_grid.best_params_}")

model_final = model_grid.best_estimator_
accuracy = model_final.score(x_test, y_test)

print(f"Accuracy: %{accuracy * 100:.2f}")

with open('./model/model_svm.pickle','wb') as f:
    pickle.dump(model_final, f)

print("SVM Model saved successfully to './model/model_svm.pickle'.")