from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from functools import lru_cache


class ImageDataset:
    def __init__(self, filenames: list[str], path2image: str, n_classes: int = 15)-> None:
        self.filenames = filenames
        self.path2image = Path(path2image)
        self.N = n_classes
        
    def __len__(self):
        return len(self.filenames)
    
    def create_path(self, indx: int)-> str:
        return str(self.path2image / self.filenames[indx])
        
    def __getitem__(self, indx):
        img = cv2.imread(self.create_path(indx))  # img должен быть изображением
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Если нужно, преобразуйте в RGB
        sample = {'image': img}
        return sample
    
    def resize(self, img, scale=2):    
        shape = (np.array(img.shape[:2])/scale).astype(np.int32)
        return cv2.resize(img,shape)
    
    
class TorchImageDataset(ImageDataset, Dataset):
    def __init__(self, filenames: list[str], path2image, imgsz=256, MEAN = (0.485, 0.456, 0.406), STD = (0.229, 0.224, 0.225)):
        super().__init__(filenames, path2image)
        self.imgsz = imgsz
        self.fransform_img= transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((self.imgsz, self.imgsz)),
                    transforms.Normalize(mean=MEAN, std=STD),
        ])
            
    @lru_cache(10000)
    def __getitem__(self, indx):
        sample = super().__getitem__(indx)
        img = sample['image']
        if img is not None:
            img = self.fransform_img(img)
            return {'image': img}