#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from EEGNET import EEGNet
from DeepConvNet import DeepConvNet
from utils import read_bci_data, train_nn, plot_results


# In[2]:


# histories
history_eeg = [None, None, None]
history_deep = [None, None, None]

# training settings
batch_size =128
epochs = 800
loss_func = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


# load data
train_data, train_label, test_data, test_label = read_bci_data()

train_dataset = TensorDataset(torch.Tensor(train_data),
                              torch.Tensor(train_label))
test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
)


# # EEGNet

# In[24]:


lr = 0.001
dropout = 0.5


# In[5]:


eeg_net_relu = EEGNet(activation='ReLU', dropout=dropout)
eeg_net_relu = eeg_net_relu.to(device)
history = train_nn(model=eeg_net_relu, epochs=epochs,
                   optimizer=optim.Adam(eeg_net_relu.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_eeg[0] = history


# In[25]:


eeg_net_leaky_relu = EEGNet(activation='LeakyReLU', dropout=dropout)
eeg_net_leaky_relu = eeg_net_leaky_relu.to(device)
history = train_nn(model=eeg_net_leaky_relu, epochs=epochs,
                   optimizer=optim.Adam(eeg_net_leaky_relu.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_eeg[1] = history


# In[32]:


eeg_net_elu = EEGNet(activation='ELU', dropout=dropout)
eeg_net_elu = eeg_net_elu.to(device)
history = train_nn(model=eeg_net_elu, epochs=epochs,
                   optimizer=optim.Adam(eeg_net_elu.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_eeg[2] = history


# In[112]:


plot_results(history_eeg, ['relu', 'leaky_relu', 'elu'], "Activation function comparison (EEGNet)")


# In[9]:


print(eeg_net_relu)


# In[22]:


save_file_name = 'eeg_net_relu.pkl'
# torch.save(eeg_net_relu.state_dict(), save_file_name)


# In[26]:


save_file_name = 'eeg_net_leaky_relu.pkl'
# torch.save(eeg_net_leaky_relu.state_dict(), save_file_name)


# In[34]:


save_file_name = 'eeg_net_elu.pkl'
# torch.save(eeg_net_elu.state_dict(), save_file_name)


# # DeepConvNet

# In[80]:


lr = 0.001
dropout = 0.5


# In[81]:


deep_conv_net_relu = DeepConvNet(activation='ReLU', dropout=dropout)
deep_conv_net_relu = deep_conv_net_relu.to(device)
history = train_nn(model=deep_conv_net_relu, epochs=epochs,
                   optimizer=optim.Adam(deep_conv_net_relu.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_deep[0] = history


# In[82]:


deep_conv_net_leaky_relu = DeepConvNet(activation='LeakyReLU', dropout=dropout)
deep_conv_net_leaky_relu = deep_conv_net_leaky_relu.to(device)
history = train_nn(model=deep_conv_net_leaky_relu, epochs=epochs,
                   optimizer=optim.Adam(deep_conv_net_leaky_relu.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_deep[1] = history


# In[83]:


deep_conv_net_elu = DeepConvNet(activation='ELU', dropout=dropout)
deep_conv_net_elu = deep_conv_net_elu.to(device)
history = train_nn(model=deep_conv_net_elu, epochs=epochs,
                   optimizer=optim.Adam(deep_conv_net_elu.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_deep[2] = history


# In[111]:


plot_results(history_deep, ['relu', 'leaky_relu', 'elu'], "Activation function comparison (DeepConvNet)")


# In[85]:


print(deep_conv_net_relu)


# In[89]:


save_file_name = 'deep_conv_net_relu.pkl'
# torch.save(deep_conv_net_relu.state_dict(), save_file_name)


# In[90]:


save_file_name = 'deep_conv_net_leaky_relu.pkl'
# torch.save(deep_conv_net_leaky_relu.state_dict(), save_file_name)


# In[91]:


save_file_name = 'deep_conv_net_elu.pkl'
# torch.save(deep_conv_net_elu.state_dict(), save_file_name)


# # Discussions

# ## Dropout Ratio

# In[102]:


lr = 0.001
history_eeg_drop = [None, history_eeg[0], None]
history_deep_drop = [None, history_deep[0], None]


# ### EEGNet 0.25

# In[103]:


eeg_net_relu_drop_25 = EEGNet(activation='ReLU', dropout=0.25)
eeg_net_relu_drop_25 = eeg_net_relu_drop_25.to(device)
history = train_nn(model=eeg_net_relu_drop_25, epochs=epochs,
                   optimizer=optim.Adam(eeg_net_relu_drop_25.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_eeg_drop[0] = history


# ### EEGNet 0.75

# In[104]:


eeg_net_relu_drop_75 = EEGNet(activation='ReLU', dropout=0.75)
eeg_net_relu_drop_75 = eeg_net_relu_drop_75.to(device)
history = train_nn(model=eeg_net_relu_drop_75, epochs=epochs,
                   optimizer=optim.Adam(eeg_net_relu_drop_75.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_eeg_drop[2] = history


# ## plot

# In[110]:


plot_results(history_eeg_drop, ['0.25', '0.5', '0.75'], "Dropout ratio comparison (EEGNet with ReLU)")


# ### DeepConvNet 0.25

# In[106]:


deep_conv_net_relu_drop_25 = DeepConvNet(activation='ReLU', dropout=0.25)
deep_conv_net_relu_drop_25 = deep_conv_net_relu_drop_25.to(device)
history = train_nn(model=deep_conv_net_relu_drop_25, epochs=epochs,
                   optimizer=optim.Adam(deep_conv_net_relu_drop_25.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_deep_drop[0] = history


# ### DeepConvNet 0.75

# In[107]:


deep_conv_net_relu_drop_75 = DeepConvNet(activation='ReLU', dropout=0.75)
deep_conv_net_relu_drop_75 = deep_conv_net_relu_drop_75.to(device)
history = train_nn(model=deep_conv_net_relu_drop_75, epochs=epochs,
                   optimizer=optim.Adam(deep_conv_net_relu_drop_75.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
history_deep_drop[2] = history


# ### plot

# In[109]:


plot_results(history_deep_drop, ['0.25', '0.5', '0.75'], "Dropout ratio comparison (DeepConvNet with ReLU)")


# ## Learning Rate

# In[113]:


history_eeg_lr = [None, None, None, history_eeg[0]]
dropout = 0.5
lrs = [0.1, 0.01, 0.0001]


# In[114]:


for i, lr in enumerate(lrs):
    eeg_lr = EEGNet(activation='ReLU', dropout=dropout).to(device)
    history = train_nn(model=eeg_lr, epochs=epochs,
                   optimizer=optim.Adam(eeg_lr.parameters(), lr=lr),
                   loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, device=device)
    history_eeg_lr[i] = history
    
history_eeg_lr[2], history_eeg_lr[3] = history_eeg_lr[3], history_eeg_lr[2]


# In[115]:


plot_results(history_eeg_lr, ['0.1', '0.01', '0.001', '0.0001'], "Learning rate comparison (EEGNet with ReLU)")

