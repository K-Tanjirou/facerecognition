# https://github.com/bubbliiiing/facenet-pytorch
import os
import argparse
import collections
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Function, Variable
from torchvision.datasets import ImageFolder

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training


def letterbox_image(image, size):
    image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(LFWDataset, self).__init__(dir, transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_paths(dir)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self, lfw_dir, file_ext="jpg"):

        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
            # for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def __getitem__(self, index):
        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = Image.open(path_1), Image.open(path_2)
        img1 = letterbox_image(img1, [self.image_size[1], self.image_size[0]])
        img2 = letterbox_image(img2, [self.image_size[1], self.image_size[0]])

        img1, img2 = np.array(img1) / 255, np.array(img2) / 255
        img1 = np.transpose(img1, [2, 0, 1])
        img2 = np.transpose(img2, [2, 0, 1])

        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)


def test(test_loader, model):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p = data_a.to(device), data_p.to(device)
            data_a, data_p, label = Variable(data_a), \
                                    Variable(data_p), Variable(label)
            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))

        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name="roc_test.png")


def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, distances,
        labels, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, distances,
        labels, 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far, best_thresholds


def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)


if __name__ == "__main__":

    input_shape = [160, 160, 3]

    cuda = True
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    batch_size = 40
    log_interval = 1

    # 数据集读取器
    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir="/home/khp/pycharm_project/FaceRecognition/lfw/lfw/", pairs_path="/home/khp/pycharm_project/FaceRecognition/lfw/pairs.txt", image_size=input_shape), batch_size=batch_size,
        shuffle=False)

    # 读取模型参数
    model = InceptionResnetV1(classify=False).to(device)
    
    state_path = '/home/khp/pycharm_project/FaceRecognition/pre_model/vggface2.pt'
    model_dict = model.state_dict()
    state_dict = torch.load(state_path)
    state_dict = {k: v for k, v in state_dict.items() if
                  (k in model_dict.keys()) and (np.shape(model_dict[k]) == np.shape(v))}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model = model.eval()

    if cuda:
        # model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.to(device)

    test(test_loader, model)
