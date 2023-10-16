import torch
import torch.nn as nn
import torchvision.models as models

class SkinNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinNet, self).__init__()
        self.network = models.vgg19(pretrained=False)  # Set pretrained to False
        # Define a custom classifier
        self.custom_classifier = nn.Sequential(
            nn.Linear(25088, 4096),  # The input size depends on the VGG19 architecture
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.custom_classifier(x)
        return x
