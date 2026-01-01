
import torch
import torch.nn as nn
import torchvision.models as models

class AccentCNN(nn.Module):
    """ResNet18-based CNN for English vs Non-English accent classification"""
    def __init__(self, dropout=0.7, freeze_layers=True):
        super(AccentCNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        if freeze_layers:
            for param in self.resnet.conv1.parameters():
                param.requires_grad = False
            for param in self.resnet.bn1.parameters():
                param.requires_grad = False
            for param in self.resnet.layer1.parameters():
                param.requires_grad = False
            for param in self.resnet.layer2.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 2)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Usage:
# model = AccentCNN(dropout=0.7, freeze_layers=True)
# model.load_state_dict(torch.load('accent_classifier_weights.pth'))
# model.eval()
