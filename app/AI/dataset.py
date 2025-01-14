import pandas as pd 
import cv2
from pathlib import Path
import numpy as np
import  re
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from functools import lru_cache


class ImageDataset:
    def __init__(self, path2csv, path2image, n_classes=15):
        # Загрузка данных из CSV
        self.df = pd.read_csv(path2csv)
        self.path2image = Path(path2image)
        self.N = n_classes 
        print('df shape:',self.df.shape)
    def __len__(self):
        return len(self.df)

    def _encode_label(self, classes):
        label = np.zeros(self.N)
        for c in np.array(classes, dtype=np.int16) - 1:
            label[c] = 1
        return label
        
    def sample(self, clss=1, count=15):
        files = []
        while len(files) < count:
            indx = np.random.randint(0, len(self.df))
            labels = self.df.iloc[indx]["OUTPUT:classes"]
            try:
                if clss in self.parse(labels):
                    files.append(self.path2image / self.file_name(indx))
            except:
                continue
        return files
    
    def create_path(self, indx):
        return str(self.path2image / self.file_name(indx))
        
    def __getitem__(self, indx):
        img = cv2.imread(self.create_path(indx))
        clases = self.parse(self.df.iloc[indx]["OUTPUT:classes"])
        label = self._encode_label(clases)
        sample = {'image': img,
                  'label': label}
        return sample
            
    def file_name(self, indx):
        return self.df.iloc[indx]['file_name']
    
    def resize(self, img, scale=2):    
        shape = (np.array(img.shape[:2]) / scale).astype(np.int32)
        return cv2.resize(img, shape)
    
    def parse(self, string):        
        # Преобразуем строку вида '[2, 5, 11]' в список [2, 5, 11]
        return [int(x) for x in re.findall("\d+", string)]
    
class FilterImageDataset(ImageDataset):
    def __init__(self, path2csv, path2image, n_classes=15, used_classes = []):
        """
            used_classes: список классов используемых для обучения модели
        """
        super().__init__(path2csv, path2image, n_classes)
        self.used_classes = set(used_classes)

    def parse(self, string):
        """
            Фильтрует нужные классы.
        """
        cls = set([int(x) for x in re.findall("\d+", string)])
        return list(cls.intersection(self.used_classes))


class TorchImageDataset(ImageDataset, Dataset):
    def __init__(self, path2csv, path2image, n_classes, imgsz=256):
        super().__init__(path2csv, path2image, n_classes)
        self.imgsz = imgsz
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)
        self.fransform_img= transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((self.imgsz, self.imgsz)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),    
                    transforms.ColorJitter(brightness=0.2, 
                                                   contrast=0.2, 
                                                   saturation=0.2, 
                                                   hue=0.2),
                    transforms.Normalize(mean=MEAN, std=STD),
        ])
            
    @lru_cache(15000)
    def __getitem__(self,indx):
        img, label = super().__getitem__(indx).values()
        if img is not None:
            img = self.fransform_img(img)
            return {'image':img,'label':label}
        else: 
            return None
        

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():  
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device       

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            if b is None:
                return None
            img,label = tuple(b.values())
            img = to_device(img,self.device)
            label = to_device(label, self.device)
            yield {"image":img,"label":label}

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    

