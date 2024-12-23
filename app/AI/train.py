import json
import cv2 
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
from collections import Counter
import os
import re
import logging
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import torch
import random

from model import ResNetMultilabel
from dataset import TorchImageDataset, DeviceDataLoader
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import mlflow
import itertools

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def validate_model(model, dataloader_valid, logger, device='cuda:0'):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for s in tqdm(dataloader_valid, desc='Validation'):
            if s is None:
                continue
            inputs = s['image']
            labels = s['label']
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = (outputs > 0.5).float() #выше порога
            predicted = outputs
            all_labels.extend(labels.detach().cpu().numpy())
            all_predictions.extend(predicted.detach().cpu().numpy())
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    f_score = f1_score(all_predictions, all_labels, average='macro')
    # print(f"Validation Results - Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}")
    # logger.debug(f"Validation Results - Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}, F_score: {f_score:.4f}")
    logger.info(f"Validation Results -  F_score: {f_score:.4f}")
    return f_score

def train_model(name, model, dataloader_train, dataloader_valid, learningRate, num_epochs, device = 'cuda:0'):
    logger = logging.getLogger('api_log')
    logger.setLevel(logging.INFO)
    p_save = Path(os.getcwd() + f'./save/{name}')
    if not p_save.is_dir():
        p_save.mkdir(parents=True)
    file_handler = logging.FileHandler(p_save / f'log_{learningRate}.txt')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.debug(f"Start_train  lr={learningRate}")
    
    # Установка оптимизатора с текущей скоростью обучения
    # optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    # weights = torch.tensor(np.load('w.npy').reshape(-1)).to('cuda')
    # criterion = nn.CrossEntropyLoss(weights)
    criterion = nn.CrossEntropyLoss()
    # Цикл обучения
    f_score_best = -1
    #Параметры обучения для логирования
    params = {
    "lr": learningRate,
    "epochs": num_epochs,
    }
    # mlflow.set_tracking_uri("http://127.0.0.1:7575")
    mlflow.set_experiment(f"Multiclass_container") #Задаем название проекта
    run_name = f"count_class_{dataloader_train.dl.dataset.dataset.N}_1_2_7_8_11"
    with mlflow.start_run(run_name=run_name) as run: #mlflow контекст
        mlflow.log_params(params)
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            all_labels = []
            all_predictions = []

            for s in tqdm(dataloader_train, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
                if s is None:
                    continue
                inputs = s['image']
                labels = s['label']
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)  # Выходы модели
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                outputs = (outputs > 0.5).float() #выше порога
                predicted = outputs
                all_labels.extend(labels.detach().cpu().numpy())
                all_predictions.extend(predicted.detach().cpu().numpy())
                if s in [70, 90]:
                    learningRate = learningRate/2
                    optimizer = optim.Adam(model.parameters(), lr=learningRate)
            
            all_predictions = np.vstack(all_predictions)
            all_labels = np.vstack(all_labels)
            f_score = f1_score(all_predictions, all_labels,average='macro')
            
            # Средняя потеря за эпоху
            epoch_loss = running_loss / len(dataloader_train)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, F_score: {f_score:.4f}")
            
            # logger.debug(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f},  F_score: {f_score:.4f}")
            f_score_val = validate_model(model, dataloader_valid, logger) 
            #Записывает метрики в mlflow
            mlflow.log_metric("training_loss", running_loss / len(train_loader), step=epoch)
            mlflow.log_metric("f_score_train", f_score, step=epoch)
            mlflow.log_metric("f_score_val", f_score_val, step=epoch)
            #------------------------------
            #mlflow.sklearn.log_model     <- регистрация модели
            #save best model 
            if f_score_val > f_score_best:
                f_score_best = f_score_val
                torch.save(model.state_dict(),  p_save / f"{name}_best.pt")

            
    # Сохранение модели last_epoch 
    # mlflow.end_run()
    torch.save(model.state_dict(), p_save / f"{name}_last.pt")


def split_data(dataset, train_ratio=0.7, val_ratio=0.1):

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    return train_data, val_data, test_data

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    device = 'cuda'
    set_seed(42)
    BSIZE = 32
    N = 15
    dataset = TorchImageDataset("../AI/source/df_1_2_7_8_11.csv",  "../../../resize_2/")
    model = ResNetMultilabel(N, "resnet50").to(device)

    t = model(torch.rand(1,3,224,224).to(device))
    print(t)
    train_data, val_data, test_data = split_data(dataset, train_ratio=0.6, val_ratio=0.05)
    print(len(train_data), len(val_data), len(test_data))
    all_labels = []
    for i in train_data.indices:  
        labels = train_data.dataset[i]['label'] 
        all_labels.append(np.where(labels == 1)[0].tolist()) 
    all_labels = list(itertools.chain(*all_labels))
    train_class_counts = {c:0 for c in range(N)}
    for l in all_labels:
        train_class_counts[l] +=1
    
    #Веса каждого класса
    class_names = sorted(train_class_counts.keys())
    print(train_class_counts)
    class_weights = np.array([ 0.0 if train_class_counts[class_name] == 0 else 1/train_class_counts[class_name] for class_name in class_names])
    for x in class_weights:
        print("{:.3f}".format(x))
    #Веса каждого примера в датасете
    weights = np.zeros(len(train_data))
    for k, i in enumerate(train_data.indices[:1000]):
        labels = train_data.dataset[i]['label']
        sample_weights = [class_weights[class_idx] for class_idx in np.where(labels == 1)[0]]
        weights[k] = np.mean(sample_weights)
    weights[np.isnan(weights)] = 0.0
    print(len(weights))
    #---------------------------
    # weights = np.array([1 for x in range(15)])
    sampler = WeightedRandomSampler(weights, num_samples = len(weights), replacement=True)
    train_loader = DeviceDataLoader(DataLoader(train_data, batch_size=BSIZE, sampler=sampler,num_workers = 4), device=device)
    # train_loader = DeviceDataLoader(DataLoader(train_data, batch_size=BSIZE, num_workers = 4), device=device)
    val_loader   = DeviceDataLoader(DataLoader(val_data, batch_size=BSIZE,num_workers = 4), device=device)
    test_loader  = DeviceDataLoader(DataLoader(test_data, batch_size=BSIZE,num_workers = 4), device=device)

    print(len(train_loader)*BSIZE, len(val_loader)*BSIZE, len(test_loader))
    train_model('res50_weghts_cls_1_2_7_8_11', model, train_loader, val_loader, 0.005, 50)
