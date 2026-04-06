import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

def get_image_paths(directory_path):
    search_path = os.path.join(directory_path, '*.jpg')
    return glob(search_path)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")


def face_detection(image_path, haar_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(haar_path)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = img[y:y+h, x:x+w] 
        return cropped_face
    
    return None
    

def process_image_for_training(path):
    try:
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if gray.shape[0] >= 100:
            gray_resize = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
        else:
            gray_resize = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_CUBIC)

        flatten_image = gray_resize.flatten()
        return flatten_image

    except Exception as e:
        print(f"Error processing ({path}): {e}")
        return None


def load_data_paths(female_folder, male_folder):
    female_files = get_image_paths(female_folder)
    male_files = get_image_paths(male_folder)
    
    df_female = pd.DataFrame(female_files, columns=['filepath'])
    df_female['gender'] = 'female'
    
    df_male = pd.DataFrame(male_files, columns=['filepath'])
    df_male['gender'] = 'male'
    
    df = pd.concat([df_female, df_male], axis=0)
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Dataset loaded: {len(df)} images found.")
    return df


def create_training_data(df):
    data_list = []  
    labels_list = [] 

    print("Processing images for training, please wait...")
    
    for index, row in df.iterrows():
        processed_img = process_image_for_training(row['filepath'])
        
        if processed_img is not None:
            data_list.append(processed_img)
            labels_list.append(row['gender'])
    
    X = np.array(data_list)
    y = np.array(labels_list)

    X = X / 255.0
    
    print(f"Final Data Shape: {X.shape}")
    return X, y