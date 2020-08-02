import pandas as pd
from torch.utils import data
import numpy as np
from torchvision import transforms

from PIL import Image


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode

        train_transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if self.mode == 'train':
            self.transform = train_transform
        else:
            self.transform = test_transform

        print(f"> Found {len(self.img_name)} {mode}ing images...")

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        label = self.label[index]

        path = self.root + self.img_name[index] + '.jpeg'
        img = self.transform(Image.open(path))

        return img, label
