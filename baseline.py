# -*- coding: utf-8 -*-
# @Time    : 2021/6/8
# @Author  : Ke
import os
import cv2
import math
import random
import face_recognition
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

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
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

# 参数
lr = 0.05
momentum = 0.09
lr_decay = 1e-4
start_epoch = 13
epochs = 40
workers = 2
train_batch_size = 40
test_batch_size = 2
test_threshold = 0.8


# 数据集路径
train_path = '/home/khp/pycharm_project/FaceRecognition/train_set/data'
test_path = '/home/khp/pycharm_project/FaceRecognition/test_set/data'
model_save_path = '/home/khp/pycharm_project/FaceRecognition/model/'


# 自定义损失函数
class ArcLoss(nn.Module):
    def __init__(self):
        super(ArcLoss, self).__init__()
        self.fc = nn.Linear(512, classes)

    def forward(self, input, target):
        out = self.fc(input)
        self.out = out
        l_fn = torch.nn.CrossEntropyLoss()
        loss = l_fn(out, target)
        return loss


# 提取人脸并保存
def data_extract(root_path):
    c_dir = sorted(os.walk(root_path).__next__()[1])
    cropped_path = root_path + '_cropped'

    index = 0
    for c in c_dir:
        pic_path = os.path.join(root_path, c)
        pic_cropped = os.path.join(cropped_path, c)
        pic = os.walk(pic_path).__next__()[2]

        if not os.path.exists(pic_cropped):     # 若该文件夹不存在就创建
            os.mkdir(pic_cropped)

        for p in pic:
            img = os.path.join(pic_path, p)
            img_cropped = os.path.join(pic_cropped, p)
            image = face_recognition.load_image_file(img)
            face_location = face_recognition.face_locations(image, model='cnn')         # 获取人脸位置
            for location in face_location:
                top, right, bottom, left = location
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image.save(img_cropped)                 # 保存获取的人脸
                index += 1
                if index % 10 == 0:
                    print(f"{index}")
    print("Extract done!")


def train():
    model.train()
    training.pass_epoch(
        model, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )


def test(distance_metric='cosine', threshold=0.8):
    model.eval()  # 设置为测试模式

    # test_set.csv
    test_csv = os.path.join(test_path, 'test_set.csv')
    df = pd.read_csv(test_csv, index_col='No.')
    df.columns = ['ground_truth']
    df.insert(1, 'predict', np.zeros(df.shape[0]))

    classes_test = test_data.classes

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

        predict_issame = np.less(threshold, similarity)         # np.less(x1, x2) 检查x1是否小于x2
        predict = int(predict_issame[0])
        df.loc[int(classes_test[idx]), 'predict'] = predict
    print(df)

    acc = np.mean([1 if int(df.loc[i, 'ground_truth']) == int(df.loc[i, 'predict']) else 0 for i in range(1, df.shape[0]+1)])
    print('-' * 50)
    print(f"Test accuracy={acc}")


if __name__ == '__main__':
    device = torch.device('cuda:0')

    # 训练数据
    data_dir = train_path + '_cropped'
    test_dir = test_path + '_cropped'

    train_transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    test_transform = transforms.Compose([
        transforms.Resize([160, 160]),
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    train_data = datasets.ImageFolder(data_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    classes_train = len(train_data)
    img_idx = np.arange(len(train_data))
    np.random.shuffle(img_idx)

    train_loader = DataLoader(train_data, batch_size=train_batch_size, sampler=SubsetRandomSampler(img_idx))
    test_loader = DataLoader(test_data, batch_size=test_batch_size)

    # facenet预训练模型
    model = InceptionResnetV1(classify=False).to(device)
    state_path = '/home/khp/pycharm_project/FaceRecognition/pre_model/vggface2.pt'
    state_dict = torch.load(state_path)
    del state_dict['logits.weight']
    del state_dict['logits.bias']
    model.load_state_dict(state_dict)

    # 定义RMSprop优化器
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=lr_decay)
    scheduler = MultiStepLR(optimizer, [5, 10])

    loss_fn = ArcLoss().to(device)
    metrics = {'fps': training.BatchTimer(), 'acc': training.accuracy}

    # 训练模型
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    for epoch in range(start_epoch, start_epoch + epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, start_epoch + epochs))
        print('-' * 10)
        train()
        model_name = f"facenet_{epoch}.pth"
        torch.save(model, os.path.join(model_save_path, model_name))
        test(threshold=test_threshold)

    torch.cuda.empty_cache()
    writer.close()
