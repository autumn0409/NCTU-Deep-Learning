# This is the python file used to show the experimental results

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import RetinopathyLoader
from resnet import resnet18, resnet50
from utils import *


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root_path = './data/'
batch_size = 8


# In[ ]:


test_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = RetinopathyLoader(data_root_path, 'test', test_transform)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
)


# # ResNet18

# ## with pretraining

# In[ ]:


net = resnet18(True)
net.load_state_dict(torch.load('resnet18_with_pretraining.pkl'))
net.to(device)
acc, targets, preds = cal_acc(net, test_loader, device)
print(f'model: resnet18 with pretraining, acc: {acc}')


# In[ ]:


plot_confusion_matrix(targets, preds, 'ResNet18 with pretraining')


# ## w/o pretraining

# In[ ]:


net = resnet18(False)
net.load_state_dict(torch.load('resnet18_without_pretraining.pkl'))
net.to(device)
acc, targets, preds = cal_acc(net, test_loader, device)
print(f'model: resnet18 w/o pretraining, acc: {acc}')


# In[ ]:


plot_confusion_matrix(targets, preds, 'ResNet18 without pretraining')


# ## comparison figure

# In[ ]:


plot_results(18, [True, False], load_history(18))


# # ResNet50

# ## with pretraining

# In[ ]:


net = resnet50(True)
net.load_state_dict(torch.load('resnet50_with_pretraining.pkl'))
net.to(device)
acc, targets, preds = cal_acc(net, test_loader, device)
print(f'model: resnet50 with pretraining, acc: {acc}')


# In[ ]:


plot_confusion_matrix(targets, preds, 'ResNet50 with pretraining')


# ## w/o pretraining

# In[ ]:


net = resnet50(False)
net.load_state_dict(torch.load('resnet50_without_pretraining.pkl'))
net.to(device)
acc, targets, preds = cal_acc(net, test_loader, device)
print(f'model: resnet50 w/o pretraining, acc: {acc}')


# In[ ]:


plot_confusion_matrix(targets, preds, 'ResNet50 without pretraining')


# ## comparison figure

# In[ ]:

h = load_history(50)
del h[2]
plot_results(50, [True, False], h)


# # Discussion

# In[ ]:


def plot_results_aug(aug, history):
    plt.figure(figsize=(10, 5))

    for history, aug in zip(history, aug):
        if aug is True:
            plt.plot(history['train_acc'], label='Train(with augmentation)')
            plt.plot(history['test_acc'], label='Test(with augmentation)')
        else:
            plt.plot(history['train_acc'], label='Train(w/o augmentation)')
            plt.plot(history['test_acc'], label='Test(w/o augmentation)')

    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')


# In[ ]:


h = load_history(50)
del h[1]
plot_results_aug([True, False], h)


# In[ ]:




