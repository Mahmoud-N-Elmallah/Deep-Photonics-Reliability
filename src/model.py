import torch.nn as nn
from torchvision import models

class PhotonicResNet50(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 4,dropout_prob: float =0.3):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_classes=num_classes
        # for FFT input
        a = self.model.conv1
        self.model.conv1 = nn.Conv2d(input_channels, a.out_channels, 
                                     kernel_size=a.kernel_size, stride=a.stride, 
                                     padding=a.padding, bias=False)
        
        # Adaptation for output classes
        a=self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=a.in_features, out_features=1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024,self.num_classes,bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x