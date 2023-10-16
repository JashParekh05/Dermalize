import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf


image_path = "/Users/jashparekh/Desktop/Dermalize/Backend/webImage.jpg"

# Define the class names
class_names = ["Acne", "Eczema", "Atopic", "Psoriasis", "Tinea", "Vitiligo"]

# Load the TensorFlow model
model = load_model('/Users/jashparekh/Desktop/Dermalize/Backend/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')  # Replace with the correct .h5 file

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Adjust to match the input size of your model
    img = np.array(img) / 255.0
    img = img.reshape((1, 224, 224, 3))  # Add batch dimension
    return img

# Function to predict skin disease from an image
def predict_skin_disease(image_path):
    img = preprocess_image(image_path)
    with tf.device('/cpu:0'):  # Use CPU to avoid any GPU memory conflicts
        predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class_index[0]]
    return predicted_class_name

# Example usage:
if __name__ == "__main__":
    predicted_class = predict_skin_disease(image_path)
    print(f"Predicted class: {predicted_class}")
