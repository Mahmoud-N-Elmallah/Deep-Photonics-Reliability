import torch
import torch.nn as nn
from torchvision import models

class PhotonicResNet50(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 4, dropout_prob: float = 0.3):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes

        # Replace conv1 for custom input channels
        old_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(input_channels, old_conv1.out_channels, 
                                     kernel_size=old_conv1.kernel_size, stride=old_conv1.stride, 
                                     padding=old_conv1.padding, bias=False)
        
        # Smart initialization: average pretrained RGB weights into grayscale filter(s)
        # This gives the new conv1 a meaningful starting point instead of random noise
        with torch.no_grad():
            pretrained_weight = old_conv1.weight  # shape [64, 3, 7, 7]
            avg_weight = pretrained_weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            self.model.conv1.weight.copy_(avg_weight.repeat(1, input_channels, 1, 1))
        
        # Adaptation for output classes
        old_fc = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=old_fc.in_features, out_features=1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x