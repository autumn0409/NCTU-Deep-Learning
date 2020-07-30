import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']),
                                axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']),
                                 axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']),
                               axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']),
                                axis=0)

    train_label = train_label - 1
    test_label = test_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    # print(train_data.shape, train_label.shape, test_data.shape,
    #       test_label.shape)

    return train_data, train_label, test_data, test_label


def train_nn(model, epochs, optimizer, loss_func, train_loader, test_loader,
             device, verbose=True):
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    if verbose:
        pbar = tqdm(total=epochs, unit=' epochs', dynamic_ncols=True, ascii=True)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device, dtype=torch.long)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(output, dim=1)
            correct += pred.eq(target).sum().item()

        history['train_loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(100. * correct / len(train_loader.dataset))

        # test
        model.eval()
        total_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device, dtype=torch.long)

            with torch.no_grad():
                output = model(data)

            total_loss += loss_func(output, target).item()
            _, pred = torch.max(output, dim=1)
            correct += pred.eq(target).sum().item()

        history['test_loss'].append(total_loss / len(test_loader))
        history['test_acc'].append(100. * correct / len(test_loader.dataset))

        if verbose:
            pbar.set_postfix({
                'train_loss': '%.4f' % (history['train_loss'][-1]),
                'train_acc': '%.2f' % (history['train_acc'][-1]),
                'test_loss': '%.4f' % (history['test_loss'][-1]),
                'test_acc': '%.2f' % (history['test_acc'][-1])
            })
            pbar.update()

    if verbose:
        pbar.close()

    return history

def plot_results(historys, activations, title):
    plt.figure(figsize=(10,5))
    for history, activation in zip(historys, activations):
        plt.plot(history['train_acc'], label=f'{activation}_train')
        plt.plot(history['test_acc'], label=f'{activation}_test')

    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title(title)

def test_acc(model, loss_func, test_loader, device):
    model.eval()
    loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)

        with torch.no_grad():
            output = model(data)

        loss += loss_func(output, target).item()
        _, pred = torch.max(output, dim=1)
        correct += pred.eq(target).sum().item()
        
    acc = 100. * correct / len(test_loader.dataset)
    return acc