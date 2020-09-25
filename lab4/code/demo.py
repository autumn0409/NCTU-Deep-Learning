#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

from utils import eval_model_gaussian, eval_model_bleu
from models import CVAE


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


epoch = 303
cvae = torch.load(f'./model_weights/cyclical_epoch{epoch}.pkl')
cvae.to(device);


# In[4]:


eval_model_bleu(cvae, True)


# In[5]:


eval_model_gaussian(cvae, True)

