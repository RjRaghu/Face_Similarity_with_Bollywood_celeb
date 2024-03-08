import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image

# Load the list of pre-computed features
feature_list = np.array(pickle.load(open('efficientnet_features.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize MTCNN face detector
detector = MTCNN()

# Load pre-trained EfficientNetB0 model with pre-trained ImageNet weights
model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg',)

# Function to preprocess and extract features from a face image
def preprocess_and_extract_features(image):
    # Detect faces in the image
    results = detector.detect_faces(image)
    
    # If no face is detected, return None
    if not results:
        return None
    
    # Extract the bounding box of the first detected face
    x, y, width, height = results[0]['box']
    
    # Crop and resize the face region to 224x224
    face = image[y:y+height, x:x+width]
    face_image = cv2.resize(face, (224, 224))
    
    # Preprocess the face image
    face_array = preprocess_input(np.expand_dims(face_image, axis=0))
    
    # Extract features using the pre-trained EfficientNetB0 model
    features = model.predict(face_array)
    return features

# Function to find the top K most similar images in the dataset
def find_most_similar_images(input_features, feature_list, top_k=5):
    similarities = cosine_similarity(input_features, feature_list)
    most_similar_indices = np.argsort(similarities.flatten())[::-1][:top_k]
    return most_similar_indices

# Load the input image
sample_img_path = 'Sample/Aftab_Shivdasani.37.jpg'
sample_img = cv2.imread(sample_img_path)

# Preprocess and extract features from the input image
input_features = preprocess_and_extract_features(sample_img)

if input_features is not None:
    # Find the top 5 most similar images in the dataset
    top_5_similar_indices = find_most_similar_images(input_features, feature_list, top_k=5)
    
    # Load and display the top 5 most similar images
    for idx in top_5_similar_indices:
        most_similar_img = cv2.imread(filenames[idx])
        print(filenames[idx])
        cv2.imshow('Similar Image', most_similar_img)
        cv2.waitKey(0)
else:
    print("No face detected in the input image.")
