import torch
from AI.device_dataloader import DeviceDataLoader
from AI.image_dataset import TorchImageDataset
from torch.utils.data import DataLoader
from AI.model import ModelMultilabel
import os


def analize_folder(sourceImg, folder):
    # Инициализируем датасет с изображениями
    dataset = TorchImageDataset(sourceImg,  folder)
    
    # Определяем устройство для вычислений (CUDA если доступно, иначе CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Оборачиваем DataLoader для перемещения данных на устройство
    dataloader = DeviceDataLoader(DataLoader(dataset, batch_size=8), device=device)
    
    # Загружаем обученную модель
    model = torch.load("ContainerSberProj/app/model.pth", map_location=torch.device('cpu'))
    model.eval()  # Переводим модель в режим оценки
    all_predictions = []  # Список для хранения предсказаний

    with torch.no_grad():  # Отключаем вычисление градиентов
        for batch in dataloader:
            images = batch['image']  # Извлекаем изображения из батча
            outputs = model(images)  # Получаем предсказания модели
            outputs = (outputs > 0.5).float()  # Преобразуем в бинарные метки
            all_predictions.append(outputs.cpu().numpy())  # Перемещаем данные на CPU и добавляем в список
            
    return(outputs)  # Выводим результаты
