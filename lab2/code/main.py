import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from EEGNET import EEGNet
from DeepConvNet import DeepConvNet
from utils import read_bci_data, train_nn, plot_results, test_acc

# training settings
lr = 0.001
batch_size = 128
epochs = 800
loss_func = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
activations = ['ReLU', 'LeakyReLU', 'ELU']

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

# EEGNet
history_eeg = []

for activation in activations:
    eeg_net = EEGNet(activation=activation, dropout=0.5).to(device)
    history = train_nn(model=eeg_net,
                       epochs=epochs,
                       optimizer=optim.Adam(eeg_net.parameters(), lr=lr),
                       loss_func=loss_func,
                       train_loader=train_loader,
                       test_loader=test_loader,
                       device=device)
    history_eeg.append(history)

    #testing
    acc = test_acc(eeg_net, loss_func, test_loader, device)
    print(f'Accuracy of EEGNet with {activation}: {acc}')

# plot results
plot_results(history_eeg, activations,
             "Activation function comparison (EEGNet)")

# DeepConvNet
history_deep = []

for activation in activations:
    deep_conv_net = DeepConvNet(activation=activation, dropout=0.5).to(device)
    history = train_nn(model=deep_conv_net,
                       epochs=epochs,
                       optimizer=optim.Adam(deep_conv_net.parameters(), lr=lr),
                       loss_func=loss_func,
                       train_loader=train_loader,
                       test_loader=test_loader,
                       device=device)
    history_deep.append(history)

    #testing
    acc = test_acc(deep_conv_net, loss_func, test_loader, device)
    print(f'Accuracy of DeepConvNet with {activation}: {acc}')

# plot results
plot_results(history_deep, activations,
             "Activation function comparison (DeepConvNet)")
