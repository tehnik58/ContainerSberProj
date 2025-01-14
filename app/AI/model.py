from torch import nn
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import timm

class MultiLabelClassifier(nn.Module): 
    def __init__(self, n_classes): 
        super(MultiLabelClassifier, self).__init__() 
        self.n_classes = n_classes 
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)  # Выберите подходящую модель ViT 
 
        self.model.head = nn.Identity() 
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.model.num_features, self.n_classes)  
         
    def forward(self, x): 
        x = self.model(x) 
        self.drop(x)
        x = self.fc(x) 
        x = F.sigmoid(x)
        return x

class ModelMultilabel(nn.Module):
    """Модель для многоклассовой классификации с использованием EfficientNet"""
    
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes  # Количество классов        
        self.model = None
        self.model = self.__efficientnet_b7()  # Инициализация модели

    def __efficientnet_b5(self):
        """Инициализирует модель EfficientNetB5 с количеством классов"""
        model = models.efficientnet_b5(pretrained=True)  # Загружаем предобученную модель
        model.classifier[1] = nn.Linear(2048, self.n_classes)  # Меняем последний слой под наше количество классов
        return model
    
    def __efficientnet_b7(self):
        """Инициализирует модель EfficientNetB7 с количеством классов"""
        model = models.efficientnet_b7(pretrained=True)  # Загружаем предобученную модель
        model.classifier[1] = nn.Linear(2560, self.n_classes)  # Меняем последний слой под наше количество классов
        return model
    def forward(self, input):
        """Прямой проход через модель"""
        out = self.model(input)  # Прогоняем вход через модель
        out = F.sigmoid(out)  # Применяем сигмоиду для многоклассовой классификации
        return out

class ResNetMultilabel(ModelMultilabel):
    def __init__(self, n_classes, model_name):
        super().__init__(n_classes)
        self.model_name = model_name
        self.model = self.__efficientnet_b5()
        
    def __efficientnet_b5(self):
        
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Identity()
            model.fc = nn.Linear(2048, self.n_classes) 
        elif self.model_name == "resnet152":
            model = models.resnet152(pretrained=True)
            model.fc = nn.Identity()
            model.fc = nn.Linear(2048, self.n_classes)
        else:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Identity()
            model.fc = nn.Linear(512, self.n_classes) 
            
        return model