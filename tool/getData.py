import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def balanceData(train_images_label):
    count = np.zeros(5)
    for i in train_images_label:
        if i == 0:
            count[0] += 1
        if i == 1:
            count[1] += 1
        if i == 2:
            count[2] += 1
        if i == 3:
            count[3] += 1
        if i == 4:
            count[4] += 1
    classWeight = [count.sum() / count[j] for j in range(5)]
    samples_weight = torch.tensor([classWeight[t] for t in train_images_label])
    samper = WeightedRandomSampler(samples_weight, len(train_images_label))
    return samper

def balanceDataBin(train_images_label):
    count = np.zeros(2)
    for i in train_images_label:
        if i == 0:
            count[0] += 1
        if i == 1:
            count[1] += 1
    classWeight = [count.sum() / count[j] for j in range(2)]
    samples_weight = torch.tensor([classWeight[t] for t in train_images_label])
    samper = WeightedRandomSampler(samples_weight, len(train_images_label))
    return samper

def getTrainDataSet(train_images_path, train_images_label, dataset_name):
    if dataset_name == 'MESSIDOR':
        data_transform = {
            "train": transforms.Compose([transforms.Resize((896, 896)),
                                         transforms.RandomRotation(180),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.499, 0.235, 0.079], [0.317, 0.16, 0.069])
                                         ])}
    if dataset_name == 'APTOS':
        data_transform = {
            "train": transforms.Compose([transforms.Resize((896, 896)),
                                         transforms.RandomRotation(180),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.460, 0.246, 0.080], [0.249, 0.138, 0.081])
                                         ])}
    dataSet = MyDataSet(train_images_path, train_images_label, transform=data_transform["train"])
    return dataSet


def getValDataSet(val_images_path, val_images_label, dataset_name):
    if dataset_name == 'MESSIDOR':
        data_transform = {
            "val": transforms.Compose([transforms.Resize((896, 896)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.499, 0.235, 0.079], [0.317, 0.16, 0.069])
                                       ])}
    if dataset_name == 'APTOS':
        data_transform = {
            "val": transforms.Compose([transforms.Resize((896, 896)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.460, 0.246, 0.080], [0.249, 0.138, 0.081])
                                       ])}
    dataSet = MyDataSet(val_images_path, val_images_label, transform=data_transform["val"])
    return dataSet

# MESSIDOR
# normMean = [0.49906003, 0.23546189, 0.07863852]
# normStd = [0.3172051, 0.16039355, 0.06926141]
# APTOS
# normMean = [0.46026966, 0.24639106, 0.08004203]
# normStd = [0.24897005, 0.13841884, 0.080653206]

