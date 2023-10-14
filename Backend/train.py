import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# Define data directories
train_dir = '/Users/jashparekh/Documents/GitHub/Dermalize/Backend/dermnet/train'
test_dir = '/Users/jashparekh/Documents/GitHub/Dermalize/Backend/dermnet/test'

# Hyperparameters
num_classes = 23
learning_rate = 0.001
batch_size = 30  # Increased batch size
num_epochs = 2  # Increased the number of epochs

# Image transforms with data augmentation
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),  # Random crop for data augmentation
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Create an instance of the ResNet model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final fully connected layer

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduling
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

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
        scheduler.step()  # Adjust learning rate

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
