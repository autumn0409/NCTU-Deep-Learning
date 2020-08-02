import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import RetinopathyLoader
from resnet import resnet18, resnet50
from utils import train_nn, load_history, save_history

# setup
batch_size = 8
epochs_18 = 10
epochs_50 = 10
loss_func = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_root_path = './data/'

# load data
train_dataset = RetinopathyLoader(data_root_path, 'train')
test_dataset = RetinopathyLoader(data_root_path, 'test')

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
)


# ResNet50 with pretraining
model = resnet50(pretrained=True)
model.to(device)

print('Start training ResNet50 with pretraining...')
history = train_nn('resnet50_with_pretraining', model, epochs_50,
                   optim.SGD(model.parameters(), lr=1e-3,
                             momentum=0.9, weight_decay=5e-4),
                   loss_func, train_loader, test_loader, device)

resnet50_history = load_history(50)
resnet50_history[0] = history
save_history(50, resnet50_history)


# ResNet18 with pretraining
model = resnet18(pretrained=True)
model.to(device)

print('Start training ResNet18 with pretraining...')
history = train_nn('resnet18_with_pretraining', model, epochs_18,
                   optim.SGD(model.parameters(), lr=1e-3,
                             momentum=0.9, weight_decay=5e-4),
                   loss_func, train_loader, test_loader, device)

resnet18_history = load_history(18)
resnet18_history[0] = history
save_history(18, resnet18_history)


# ResNet50 without pretraining
model = resnet50(pretrained=False)
model.to(device)

print('Start training ResNet50 without pretraining...')
history = train_nn('resnet50_without_pretraining', model, epochs_50,
                   optim.SGD(model.parameters(), lr=1e-3,
                             momentum=0.9, weight_decay=5e-4),
                   loss_func, train_loader, test_loader, device)

resnet50_history = load_history(50)
resnet50_history[1] = history
save_history(50, resnet50_history)


# ResNet18 without pretraining
model = resnet18(pretrained=False)
model.to(device)

print('Start training ResNet18 without pretraining...')
history = train_nn('resnet18_without_pretraining', model, epochs_18,
                   optim.SGD(model.parameters(), lr=1e-3,
                             momentum=0.9, weight_decay=5e-4),
                   loss_func, train_loader, test_loader, device)

resnet18_history = load_history(18)
resnet18_history[1] = history
save_history(18, resnet18_history)
