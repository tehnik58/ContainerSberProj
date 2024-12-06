from torch import nn
from torchvision import models
import torch.nn.functional as F


class ModelMultilabel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes        
        self.model = None
        self.model = self.__efficientnet_b5()
        
    def __efficientnet_b5(self):
        model = models.efficientnet_b5(pretrained=True)
        model.classifier[1] = nn.Linear(2048,self.n_classes)
        return model
    
    def forward(self, input):
        out = self.model(input)
        out = F.sigmoid(out)
        return out