import torch.nn as nn


class DeepConvNet(nn.Module):
    def __init__(self, activation, dropout=0.5):
        super(DeepConvNet, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(1, 25, kernel_size=(1, 5)),
                                    nn.Conv2d(25, 25, kernel_size=(2, 1)),
                                    nn.BatchNorm2d(25),
                                    self.activation(activation),
                                    nn.MaxPool2d(kernel_size=(1, 2)),
                                    nn.Dropout(p=dropout))
        self.conv_2 = nn.Sequential(nn.Conv2d(25, 50, kernel_size=(1, 5)),
                                    nn.BatchNorm2d(50),
                                    self.activation(activation),
                                    nn.MaxPool2d(kernel_size=(1, 2)),
                                    nn.Dropout(p=dropout))
        self.conv_3 = nn.Sequential(nn.Conv2d(50, 100, kernel_size=(1, 5)),
                                    nn.BatchNorm2d(100),
                                    self.activation(activation),
                                    nn.MaxPool2d(kernel_size=(1, 2)),
                                    nn.Dropout(p=dropout))
        self.conv_4 = nn.Sequential(nn.Conv2d(100, 200, kernel_size=(1, 5)),
                                    nn.BatchNorm2d(200),
                                    self.activation(activation),
                                    nn.MaxPool2d(kernel_size=(1, 2)),
                                    nn.Dropout(p=dropout))
        self.classify = nn.Sequential(
            nn.Linear(in_features=8600, out_features=2, bias=True))

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = out.view(x.shape[0], -1)
        out = self.classify(out)
        return out

    def activation(self, activation):
        if activation == 'ReLU':
            return nn.ReLU()
        elif activation == 'LeakyReLU':
            return nn.LeakyReLU()
        elif activation == 'ELU':
            return nn.ELU(alpha=1.0)
        else:
            print('Unknown activation function.')
            exit(1)
