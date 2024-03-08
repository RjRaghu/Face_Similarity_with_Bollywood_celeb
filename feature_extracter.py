import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image

# Load filenames
filenames = []

# Replace 'path_to_your_data_folder' with the path to the folder containing your images
data_folder = 'Data'
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.jpg'):
            filenames.append(os.path.join(root, file))

# Load EfficientNetB0 model with pre-trained ImageNet weights
model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def batch_feature_extractor(img_paths, model):
    features = []
    for img_path in tqdm(img_paths):
        img_array = preprocess_image(img_path)
        img_array = np.expand_dims(img_array, axis=0)
        features.append(model.predict(img_array)[0])
    return features

# Extract features for all images
all_features = batch_feature_extractor(filenames, model)

# Save features
pickle.dump(all_features, open('efficientnet_features.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
