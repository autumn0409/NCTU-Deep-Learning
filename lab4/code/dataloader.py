import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import rnn

import numpy as np
import random

from utils import TENSES
from utils import word2tensor, tense2tensor


def getData():
    words = []
    tenses = []

    with open('train.txt') as f:
        for line in f:
            words.append(line.split('\n')[0].split(' '))
            tenses.append([t for t in TENSES])

    return np.array(words), np.array(tenses)


class MyData(Dataset):
    def __init__(self):
        words, tenses = getData()
        self.words = words
        self.tenses = tenses

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        tense_index_input = random.randint(0, 4-1)
        input = self.words[idx][tense_index_input]
        condition = self.tenses[idx][tense_index_input]
        input_tensor = word2tensor(input)
        condition_tensor = tense2tensor(condition)
        return input_tensor, condition_tensor


def collate_fn(data):
    batch_size = len(data)
    input_tensor = [data[i][0] for i in range(batch_size)]
    input_cond_tensor = torch.LongTensor([data[i][1] for i in range(batch_size)])
    input_tensor = torch.LongTensor(rnn.pad_sequence(input_tensor, batch_first=True, padding_value=1))
    return input_tensor, input_cond_tensor


def get_train_loader(batch_size):
    train_dataset = MyData()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return train_loader
