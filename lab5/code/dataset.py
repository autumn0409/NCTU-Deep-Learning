import torch
from torch.utils.data import Dataset

import json
import numpy as np
from PIL import Image


def getData(mode, filename=None):
    if mode == 'train':
        with open('train.json') as f:
            train_data = json.load(f)

        img_names = np.array(list(train_data.keys()))
        objects = np.array(list(train_data.values()), dtype=object)

        return img_names, objects
    else:
        with open(filename) as f:
            test_data = json.load(f)

        return np.array(test_data, dtype=object)


class TrainingDataset(Dataset):
    def __init__(self, transform):
        self.img_root_path = './data/'
        self.img_names, self.objects = getData('train')
        self.transform = transform

        with open('objects.json') as f:
            self.objects_json = json.load(f)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        label = [0 for i in range(24)]
        for object_name in self.objects[index]:
            label[self.objects_json[object_name]] = 1
        label = torch.tensor(label, dtype=torch.float)

        path = self.img_root_path + self.img_names[index]
        img = Image.open(path)
        img = img.convert("RGB")
        img = self.transform(img)

        return img, label


class TestingDataset(Dataset):
    def __init__(self, filename):
        self.objects = getData('test', filename)

        with open('objects.json') as f:
            self.objects_json = json.load(f)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        label = [0 for i in range(24)]
        for object_name in self.objects[index]:
            label[self.objects_json[object_name]] = 1
        label = torch.tensor(label, dtype=torch.float)

        return label
