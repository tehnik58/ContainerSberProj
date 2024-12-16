from torch import nn
from torchvision import models
import torch.nn.functional as F


class ModelMultilabel(nn.Module):
    """Модель для многоклассовой классификации с использованием EfficientNet"""
    
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes  # Количество классов        
        self.model = None
        self.model = self.__efficientnet_b5()  # Инициализация модели

    def __efficientnet_b5(self):
        """Инициализирует модель EfficientNetB5 с количеством классов"""
        model = models.efficientnet_b5(pretrained=True)  # Загружаем предобученную модель
        model.classifier[1] = nn.Linear(2048, self.n_classes)  # Меняем последний слой под наше количество классов
        return model
    
    def forward(self, input):
        """Прямой проход через модель"""
        out = self.model(input)  # Прогоняем вход через модель
        out = F.sigmoid(out)  # Применяем сигмоиду для многоклассовой классификации
        return out
