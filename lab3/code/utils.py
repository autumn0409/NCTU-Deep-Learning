import torch
from sklearn.metrics import confusion_matrix

import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_nn(model_name, model, epochs, optimizer, loss_func, train_loader, test_loader, device):
    history = {
        'train_acc': [],
        'test_acc': []
    }
    highest_acc = 0

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0
        correct = 0

        pbar_train = tqdm(total=len(train_loader), unit=' batches',  ascii=True)
        pbar_train.set_description(
            "{:^10} ({}/{})".format("Training", epoch, epochs))

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(output, dim=1)
            correct += pred.eq(target).sum().item()

            pbar_train.set_postfix({
                'train_loss':
                '%.4f' % (train_loss / (batch_idx + 1)),
                'train_acc':
                '%.2f%%' % (100. * correct / ((batch_idx + 1)
                                              * train_loader.batch_size))
            })
            pbar_train.update()

        history['train_acc'].append(100. * correct / len(train_loader.dataset))
        pbar_train.close()

        # test
        model.eval()
        test_loss = 0
        correct = 0

        pbar_test = tqdm(total=len(test_loader), unit=' batches',  ascii=True)
        pbar_test.set_description(
            "{:^10} ({}/{})".format("Testing", epoch, epochs))

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output = model(data)

            test_loss += loss_func(output, target).item()
            _, pred = torch.max(output, dim=1)
            correct += pred.eq(target).sum().item()

            pbar_test.set_postfix({
                'test_loss':
                '%.4f' % (test_loss / (batch_idx + 1)),
                'test_acc':
                '%.2f%%' % (100. * correct /
                            ((batch_idx + 1) * test_loader.batch_size))
            })
            pbar_test.update()

        history['test_acc'].append(100. * correct / len(test_loader.dataset))
        if history['test_acc'][-1] > highest_acc:
            torch.save(model.state_dict(), f'{model_name}.pkl')
            highest_acc = history['test_acc'][-1]

        pbar_test.close()

    return history


def plot_results(num_layers, pretrained, history):
    plt.figure(figsize=(10, 5))

    for history, pretrained in zip(history, pretrained):
        if pretrained is True:
            plt.plot(history['train_acc'], label='Train(with pretraining)')
            plt.plot(history['test_acc'], label='Test(with pretraining)')
        else:
            plt.plot(history['train_acc'], label='Train(w/o pretraining)')
            plt.plot(history['test_acc'], label='Test(w/o pretraining)')

    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.title(f'Result Comparison (ResNet{num_layers})')


def save_history(num_layers, history):
    filename = None

    if num_layers == 18:
        filename = "resnet18_history.pkl"
    else:
        filename = "resnet50_history.pkl"

    with open(filename, "wb") as fp:
        pickle.dump(history, fp)


def load_history(num_layers):
    filename = None

    if num_layers == 18:
        filename = "resnet18_history.pkl"
    else:
        filename = "resnet50_history.pkl"

    if os.path.isfile(filename):
        with open(filename, "rb") as fp:
            return pickle.load(fp)
    else:
        return [None, None]


def cal_acc(model, test_loader, device):
    model.eval()
    correct = 0
    preds = []
    targets = []

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)

        _, pred = torch.max(output, dim=1)
        preds.extend(pred.tolist())
        targets.extend(target.tolist())
        correct += pred.eq(target).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    return acc, targets, preds


def plot_confusion_matrix(y_true, y_pred, title):
    np.set_printoptions(precision=2)
    classes = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(title.replace(" ", "_") + "_cfmatrx.png")
    plt.show()
    plt.close()
