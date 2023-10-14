import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

categories = ["Acne", "Eczema", "Atopic Dermatitis", "Psoriasis", "Tinea", "Vitiligo"]


class DataProcessor:
    def __init__(self, data_dir, categories):
        self.data_dir = data_dir
        self.categories = categories
        self.images = []
        self.labels = []

    def load_data(self):
        for category_id, category in enumerate(self.categories):
            path = os.path.join(self.data_dir, category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img))
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    img_array = cv2.resize(img_array, (180, 180))
                    self.images.append(img_array)
                    self.labels.append(category_id)
                except Exception as e:
                    pass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

class SkinDiseaseClassifier:
    def __init__(self, categories):
        self.categories = categories
        self.model = Sequential()
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(len(categories), activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

# Rest of the code remains the same

    def train_model(self, x_train, y_train, epochs):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

    def save_model(self, model_filename):
        self.model.save(model_filename)

    def load_model(self, model_filename):
        from tensorflow.keras.models import load_model
        self.model = load_model(model_filename)

class SkinDiseaseApp:
    categories = ["Acne", "Eczema", "Atopic Dermatitis", "Psoriasis", "Tinea", "Vitiligo"]

    def __init__(self, model_filename):
        self.model = SkinDiseaseClassifier(self.categories)
        self.model.load_model(model_filename)

    def predict_skin_disease(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (180, 180))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = self.model.model.predict(img)
        predicted_class_index = np.argmax(prediction)
        return self.categories[predicted_class_index]

# Assuming you define 'categories' somewhere in your code.
