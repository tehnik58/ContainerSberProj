def to_device(data, device):
    """Перемещает тензор(ы) на выбранное устройство"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]  # Рекурсивно вызываем для списка или кортежа
    return data.to(device, non_blocking=True)  # Перемещаем данные на устройство с флагом non_blocking

class DeviceDataLoader():
    """Обертка над DataLoader для перемещения данных на устройство"""
    
    def __init__(self, dl, device):
        self.dl = dl  # Обычный DataLoader
        self.device = device  # Устройство (например, CPU или CUDA)
        
    def __iter__(self):
        """Возвращает пакет данных после перемещения их на устройство"""
        for b in self.dl:
            img = b['image']  # Извлекаем изображение из батча
            img = to_device(img, self.device)  # Перемещаем изображение на выбранное устройство
            yield {"image": img}  # Отправляем на устройство батч с изображением

    def __len__(self):
        """Возвращает количество пакетов в DataLoader"""
        return len(self.dl)
