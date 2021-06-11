# -*- coding: utf-8 -*-
# @Time    : 2021/6/8
# @Author  : Ke
import os
import cv2
import random
import face_recognition
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pymongo import MongoClient

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from baseline import MTCNN, InceptionResnetV1, fixed_image_standardization, training

# 参数
lr = 0.05
momentum = 0.09
lr_decay = 1e-4
start_epoch = 10
epochs = 50
workers = 2
train_batch_size = 40
test_batch_size = 2
test_threshold = 0.9997

# 数据集路径
test_path = '/home/khp/pycharm_project/FaceRecognition/test_set/data'


def test(distance_metric='cosine', threshold=0.7):
    model.eval()  # 设置为测试模式

    # test_set.csv
    test_csv = os.path.join(test_path, 'test_set.csv')
    df = pd.read_csv(test_csv, index_col='No.')
    df.columns = ['ground_truth']
    df.insert(1, 'predict', np.zeros(df.shape[0]))

    classes_test = test_data.classes
    # print(classes_test)

    for idx, (data, _) in enumerate(test_loader):

        data = data.to(device)

        embeddings = model(data)

        embedding1 = embeddings[0].cpu().detach().numpy()
        embedding2 = embeddings[1].cpu().detach().numpy()
        if distance_metric == 'Euclidian':
            diff = np.subtract(embedding1, embedding2)
            dist = np.sum(np.square(diff), 1)
            similarity = np.floor_divide(1, 1 + dist)
        elif distance_metric == 'cosine':
            dot = np.sum(np.multiply(embedding1, embedding2))
            norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            similarity = dot / norm
            # dist = np.arccos(similarity) / math.pi
            similarity = 0.5 + 0.5 * similarity
        print(similarity)

        predict_issame = np.less(threshold, similarity)
        predict = int(predict_issame)
        df.loc[int(classes_test[idx]), 'predict'] = predict
    print(df)

    acc = np.mean([1 if int(df.loc[i, 'ground_truth']) == int(df.loc[i, 'predict']) else 0 for i in range(1, df.shape[0]+1)])
    print('-' * 50)
    print(f"Test accuracy={acc}")


if __name__ == '__main__':
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    # 训练数据
    data_dir = train_path + '_cropped'
    test_dir = test_path + '_cropped'

    test_transform = transforms.Compose([
        transforms.Resize([160, 160]),
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=test_batch_size)

    # facenet预训练模型
    # model = InceptionResnetV1(classify=False).to(device)
    model_idx = 12
    model_path = f"/home/khp/pycharm_project/FaceRecognition/model/facenet_{model_idx}.pth"
    model = torch.load(model_path)
    # model = nn.DataParallel(model)
    model = model.to(device)

    test(threshold=test_threshold)
