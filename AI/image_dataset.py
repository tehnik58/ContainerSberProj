from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from functools import lru_cache


class ImageDataset:
    """Класс для работы с набором изображений"""
    
    def __init__(self, filenames: list[str], path2image: str, n_classes: int = 15)-> None:
        self.filenames = filenames  # Список файлов с изображениями
        self.path2image = Path(path2image)  # Путь к директории с изображениями
        self.N = n_classes  # Количество классов

    def __len__(self):
        """Возвращает количество элементов в датасете"""
        return len(self.filenames)
    
    def create_path(self, indx: int)-> str:
        """Создает полный путь к изображению по индексу"""
        return str(self.path2image / self.filenames[indx])
        
    def __getitem__(self, indx):
        """Возвращает изображение по индексу"""
        img = cv2.imread(self.create_path(indx))  # Считываем изображение с диска
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # При необходимости преобразуем в RGB
        sample = {'image': img}  # Оборачиваем в словарь
        return sample
    
    def resize(self, img, scale=2):    
        """Изменяет размер изображения"""
        shape = (np.array(img.shape[:2])/scale).astype(np.int32)  # Уменьшаем размер изображения по указанному масштабу
        return cv2.resize(img, shape)  # Возвращаем измененный размер изображения
    
    
class TorchImageDataset(ImageDataset, Dataset):
    """Класс для работы с изображениями в формате Dataset для PyTorch"""
    
    def __init__(self, filenames: list[str], path2image, imgsz=256, MEAN = (0.485, 0.456, 0.406), STD = (0.229, 0.224, 0.225)):
        super().__init__(filenames, path2image)  # Инициализируем родительский класс
        self.imgsz = imgsz  # Размер изображения
        self.fransform_img = transforms.Compose([  # Композиция преобразований изображений
                    transforms.ToTensor(),  # Преобразуем в тензор
                    transforms.Resize((self.imgsz, self.imgsz)),  # Изменяем размер
                    transforms.Normalize(mean=MEAN, std=STD),  # Нормализуем изображение
        ])
            
    @lru_cache(10000)  # Кэшируем результаты для ускорения
    def __getitem__(self, indx):
        """Возвращает обработанное изображение по индексу"""
        sample = super().__getitem__(indx)  # Извлекаем исходное изображение из родительского класса
        img = sample['image']
        if img is not None:
            img = self.fransform_img(img)  # Применяем трансформацию
            return {'image': img}
