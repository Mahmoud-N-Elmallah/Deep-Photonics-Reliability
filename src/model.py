import torch.nn as nn
from torchvision import models

class PhotonicResNet50(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # for single-channel FFT input
        a = self.model.conv1
        self.model.conv1 = nn.Conv2d(input_channels, a.out_channels, 
                                     kernel_size=a.kernel_size, stride=a.stride, 
                                     padding=a.padding, bias=False)
        
        # Adaptation for output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x=self.model(x)
        return x