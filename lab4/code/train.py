import torch

import pickle

from models import CVAE
from utils import trainEpochs, plot_results
from dataloader import get_train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------Hyper Parameters---------- #
hidden_size = 256
cond_embedding_size = 8
latent_size = 32
n_epochs = 900
batch_size = 16

learning_rate = 0.001

train_loader = get_train_loader(batch_size=batch_size)


for KL_annealing_method in ['cyclical', 'monotonic']:
    cvae = CVAE(hidden_size, latent_size, cond_embedding_size).to(device)

    history = trainEpochs(train_loader=train_loader, cvae=cvae,
                          n_epochs=n_epochs, learning_rate=learning_rate,
                          KL_annealing_method=KL_annealing_method)

    plot_results(history, KL_annealing_method)

    filename = f'./histories/{KL_annealing_method}.pkl'
    with open(filename, "wb") as fp:
        pickle.dump(history, fp)
