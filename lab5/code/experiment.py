#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import TestingDataset
from evaluator import evaluation_model

import numpy as np
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = evaluation_model()


# In[ ]:


test_dataset = TestingDataset('test.json')
test_loader = torch.utils.data.DataLoader(test_dataset, 32, shuffle=False)


# In[ ]:


netG = torch.load('./models_weight/netG.pkl')


# In[ ]:


for labels in test_loader:
    z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (32, 104))))
    img = netG(z, labels.to(device))
    img = F.interpolate(img, size=64)
    print(f'Acc = {evaluator.eval(img, labels)}')
    save_image(make_grid(img * 0.5 + 0.5), './test.png')


# In[ ]:


with open('history.pkl', "rb") as fp:
    history = pickle.load(fp)
    
plt.plot(history['test_acc'])
plt.xlabel('Iters')
plt.ylabel('Accuracy')

