import torch
from device_dataloader import DeviceDataLoader
from image_dataset import TorchImageDataset
from torch.utils.data import DataLoader
from model import ModelMultilabel


if __name__ == "__main__":
    # Инициализируем датасет с изображениями
    dataset = TorchImageDataset(["1TimePhoto_20241001_070002 (2).jpg", "1TimePhoto_20241001_070002 (3).jpg"],  "../Resize/")
    
    # Определяем устройство для вычислений (CUDA если доступно, иначе CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Оборачиваем DataLoader для перемещения данных на устройство
    dataloader = DeviceDataLoader(DataLoader(dataset, batch_size=8), device=device)
    
    # Загружаем обученную модель
    model = torch.load("../model.pth")
    model.eval()  # Переводим модель в режим оценки
    all_predictions = []  # Список для хранения предсказаний

    with torch.no_grad():  # Отключаем вычисление градиентов
        for batch in dataloader:
            images = batch['image']  # Извлекаем изображения из батча
            outputs = model(images)  # Получаем предсказания модели
            outputs = (outputs > 0.5).float()  # Преобразуем в бинарные метки
            all_predictions.append(outputs.cpu().numpy())  # Перемещаем данные на CPU и добавляем в список
            
    print(outputs)  # Выводим результаты
