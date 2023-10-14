import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import SkinNet
from data_processor import SkinDiseaseApp

# Define data directories
train_dir = '/Users/jashparekh/Documents/GitHub/Dermalize/Backend/dermnet/train'
test_dir = '/Users/jashparekh/Documents/GitHub/Dermalize/Backend/dermnet/test'

# Hyperparameters
num_classes = 23
learning_rate = 0.001
batch_size = 64
num_epochs = 1



# Image transforms
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Data augmentation - random crop
    transforms.RandomHorizontalFlip(),  # Data augmentation - random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Create an instance of the SkinNet model
model = SkinNet(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1} - Loss: {running_loss / len(train_loader)}')

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'skin_model.pth')

# Validation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total}%')

# Test the model using SkinDiseaseApp
app = SkinDiseaseApp('skin_model.pth')
image_path = "/Users/jashparekh/Documents/GitHub/Dermalize/Backend/acne.jpeg"
predicted_class = app.predict_skin_disease(image_path)

predicted_class = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]

print(f'Predicted class: {predicted_class}')
