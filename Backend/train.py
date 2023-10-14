import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

# Define the class names
class_names = ["Acne", "Eczema", "Atopic", "Psoriasis", "Tinea", "Vitiligo"]

# Load your saved model
model = tf.keras.models.load_model('6claass.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = vgg_model.predict(img.reshape(1, 180, 180, 3))
    img = img.reshape(1, -1)
    return img

def predict_skin_disease(image_path):
    img = preprocess_image(image_path)
    pred = model.predict(img)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Example usage:

