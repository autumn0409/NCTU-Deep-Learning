import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, activation, dropout=0.25):
        super(EEGNet, self).__init__()

        self.firstConv = nn.Sequential(
            nn.Conv2d(1,
                      16,
                      kernel_size=(1, 51),
                      stride=(1, 1),
                      padding=(0, 25),
                      bias=False),
            nn.BatchNorm2d(16,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16,
                      32,
                      kernel_size=(2, 1),
                      stride=(1, 1),
                      groups=16,
                      bias=False),
            nn.BatchNorm2d(32,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            self.activation(activation),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32,
                      32,
                      kernel_size=(1, 15),
                      stride=(1, 1),
                      padding=(0, 7),
                      bias=False),
            nn.BatchNorm2d(32,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            self.activation(activation),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout),
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True))

    def forward(self, x):
        out = self.firstConv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = out.view(-1, 736)  # flattern
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
