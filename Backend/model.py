import torch
import torch.nn as nn
from torchvision import models

class SkinNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinNet, self).__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)  # Correct the arguments

    def forward(self, x):
        return self.network(x)
