import os
import pickle
import numpy as np
import PIL
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
