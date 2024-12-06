import torch
from device_dataloader import DeviceDataLoader
from image_dataset import TorchImageDataset
from torch.utils.data import DataLoader
from model import ModelMultilabel


if __name__ == "__main__":
    dataset = TorchImageDataset(["1TimePhoto_20241001_070002 (2).jpg", "1TimePhoto_20241001_070002 (3).jpg"],  "../Resize/")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DeviceDataLoader(DataLoader(dataset, batch_size=8), device=device)
    
    model = torch.load("../model.pth")
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image']
            outputs = model(images)
            outputs = (outputs > 0.5).float()
            all_predictions.append(outputs.cpu().numpy())
            
    print(outputs)