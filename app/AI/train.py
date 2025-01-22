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

from model import ModelMultilabel, MultiLabelClassifier, ModelBinary
from dataset import TorchImageDataset, DeviceDataLoader
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import mlflow
import itertools
from datetime import datetime
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def validate_model(model, dataloader_valid, logger, device='cuda:0'):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        flag = True
        for s in tqdm(dataloader_valid, desc='Validation'):
            if s is None:
                continue
            if s['label'] is None:
                continue

            inputs = s['image']
            labels = s['label']
            inputs, labels = inputs.to(device), labels.to(device)
            if flag:
                y_pred_1 = model(inputs)
                y_pred = (y_pred_1 > 0.5).float() #выше порога
                flag = False
                # print(y_pred_1, y_pred)
                print(y_pred[:4])
                print(labels[:4])

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

def train_model(name, model, dataloader_train, dataloader_valid, learningRate, num_epochs, device = 'cuda:0', cls_weights = None):
    logger = logging.getLogger('api_log')
    logger.setLevel(logging.INFO)
    p_save = Path(os.getcwd() + f'/save/{name}')
    if not p_save.is_dir():
        p_save.mkdir(parents=True)
    file_handler = logging.FileHandler(p_save / f'log_{learningRate}.txt')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.debug(f"Start_train  lr={learningRate}")
    
    # Установка оптимизатора с текущей скоростью обучения
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=learningRate)
    # weights = torch.tensor(np.load('w.npy').reshape(-1)).to('cuda')
    # criterion = nn.CrossEntropyLoss(weights)
    if cls_weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(torch.tensor(cls_weights).to(device))
    # Цикл обучения
    f_score_best = -1
    #Параметры обучения для логирования
    params = {
    "lr": learningRate,
    "epochs": num_epochs,
    }
    # mlflow.set_tracking_uri("http://127.0.0.1:7575")
    mlflow.set_experiment(f"Multiclass_container") #Задаем название проекта
    run_name = f"{name}"
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
                    for group in optim.param_groups:
                        group['lr'] /= 10
            
            all_predictions = np.vstack(all_predictions)
            all_labels = np.vstack(all_labels)
            f_score = f1_score(all_predictions, all_labels,average='macro')
            
            # Средняя потеря за эпоху
            epoch_loss = running_loss / len(dataloader_train)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, F_score: {f_score:.4f}")
            
            # logger.debug(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f},  F_score: {f_score:.4f}")
            f_score_val = validate_model(model, dataloader_valid, logger, device) 
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
    device = 'cuda:2'
    set_seed(42)
    BSIZE = 32
    N = 2
    type = "binary"
    path2csv = "./source/df_target_other.csv"
    dataset = TorchImageDataset(path2csv,  "../../../resize_2/", N, 224)
    if type == 'B5':
        model = ModelMultilabel(N).to(device)
    elif type == 'B7':
        model = ModelMultilabel(N, type).to(device)
    elif type == 'VIT':
        model = MultiLabelClassifier(N).to(device)
    elif type == 'binary':
        model = ModelBinary("eff_v2_m").to(device)

    t = model(torch.rand(1,3,224,224).to(device))
    print(t)
    train_data, val_data, test_data = split_data(dataset, train_ratio=0.6, val_ratio=0.07)
    print(len(train_data), len(val_data), len(test_data))
    all_labels = []
    for i in train_data.indices:  
        if train_data.dataset[i] is None:
            continue
        if not 'label' in train_data.dataset[i]:
            continue
        labels = train_data.dataset[i]['label'] 
        all_labels.append(np.where(labels == 1)[0].tolist()) 
    all_labels = list(itertools.chain(*all_labels))
    train_class_counts = {c:0 for c in range(N)}
    for l in all_labels:
        train_class_counts[l] +=1
    
    #Веса каждого класса
    class_names = sorted(train_class_counts.keys())
    print(train_class_counts)
    count_sum = np.sum(list(x[1] for x in train_class_counts.items()))
    class_weights = np.array([0.0 if train_class_counts[class_name] == 0 else train_class_counts[class_name]/count_sum for class_name in class_names])
    
    for x in class_weights:
        print("{:.3f}".format(x), " ", "{:.3f}".format(1/x),)
    
    #Веса каждого примера в датасете
    weights = np.zeros(len(train_data))
    for k, i in enumerate(train_data.indices):
        if train_data.dataset[i] is None:
            continue
        labels = train_data.dataset[i]['label']
        sample_weights = [class_weights[class_idx] for class_idx in np.where(labels == 1)[0]]
        weights[k] = np.mean(sample_weights)
    weights[np.isnan(weights)] = 0.0


    #---------------------------
    # weights = np.array([1 for x in range(15)])
    sampler = WeightedRandomSampler(weights, num_samples = len(weights), replacement=True)
    train_loader = DeviceDataLoader(DataLoader(train_data, batch_size=BSIZE, sampler=sampler,num_workers = 4), device=device)
    # train_loader = DeviceDataLoader(DataLoader(train_data, batch_size=BSIZE, num_workers = 4), device=device)
    val_loader   = DeviceDataLoader(DataLoader(val_data, batch_size=BSIZE,num_workers = 4), device=device)
    test_loader  = DeviceDataLoader(DataLoader(test_data, batch_size=BSIZE,num_workers = 4), device=device)


    # all_labels = []
    # for batch in tqdm(train_loader, desc="Evaluating"):
    #     inputs = batch['image'].to(device)
    #     labels = batch['label'].to(device)
    #     all_labels.append(labels.cpu().numpy())
    # all_labels = np.vstack(all_labels)
    # print(all_labels.sum(axis=0))
    print(len(train_loader)*BSIZE, len(val_loader)*BSIZE, len(test_loader))
    s = str(datetime.now()).split('.')[0].replace(" ","_")
    train_model(f'{type}_b5_target_other{s}', model, train_loader, val_loader, 0.005, 50, device, 1/class_weights)
