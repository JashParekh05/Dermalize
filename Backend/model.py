import torch.nn as nn
from torchvision import models

# Define the model architecture
class SkinNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinNet, self).__init__()
        # Use a pretrained ResNet model
        self.network = models.resnet50(pretrained=True)  # Use pretrained=True
        # Replace the last fully connected layer for our num_classes
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.network(x)

