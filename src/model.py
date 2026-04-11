import torch
import torch.nn as nn
from torchvision import models

class PhotonicResNet18(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 4, dropout_prob: float = 0.3):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.num_classes = num_classes

        # Replace conv1 for custom input channels
        old_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(input_channels, old_conv1.out_channels, 
                                     kernel_size=old_conv1.kernel_size, stride=old_conv1.stride, 
                                     padding=old_conv1.padding, bias=False)
        
        # Smart initialization
        with torch.no_grad():
            pretrained_weight = old_conv1.weight  
            avg_weight = pretrained_weight.mean(dim=1, keepdim=True)  
            self.model.conv1.weight.copy_(avg_weight.repeat(1, input_channels, 1, 1))
        
        # FC stack
        old_fc = self.model.fc
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=old_fc.in_features, out_features=256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, self.num_classes, bias=True)
        )

    def forward(self, x, return_attention=False):
        if not return_attention:
            return self.model(x)
        
        # Forward pass to layer4
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        feature_map = x 
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.model.fc(x)
        
        # SHARP ATTENTION:
        # Instead of just mean, we use mean + quadratic scaling to sharpen the peaks
        attention_map = torch.mean(feature_map, dim=1, keepdim=True) 
        attention_map = torch.clamp(attention_map, min=0) # ReLU
        attention_map = attention_map.pow(2) # Sharpens high-activation areas 
        
        # Min-Max Normalization per batch sample
        B, C, H, W = attention_map.shape
        flat_map = attention_map.view(B, -1)
        map_min = flat_map.min(dim=1, keepdim=True)[0]
        map_max = flat_map.max(dim=1, keepdim=True)[0]
        norm_attention = (flat_map - map_min) / (map_max - map_min + 1e-8)
        norm_attention = norm_attention.view(B, 1, H, W)
        
        return logits, norm_attention