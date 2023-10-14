import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the class names for your specific application
class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']

# Load your pre-trained model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, len(class_names))  # Modify the fully connected layer according to your model architecture

# Load the saved model checkpoint (make sure to use the correct file path)
checkpoint = torch.load('skin_model_updated.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    image_path = "temp.jpg"  # Save the image temporarily
    image.save(image_path)

    input_tensor = preprocess_image(image_path)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class label
    _, predicted_class = output.max(1)
    class_label = class_names[predicted_class]

    return jsonify({'class_label': class_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
