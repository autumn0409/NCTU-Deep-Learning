import torch.nn as nn
import torchvision


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, reduced_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(in_channels, reduced_channels)
        self.bn1 = nn.BatchNorm2d(reduced_channels)

        self.conv2 = conv3x3(reduced_channels, reduced_channels, stride)
        self.bn2 = nn.BatchNorm2d(reduced_channels)

        out_channels = reduced_channels * self.expansion
        self.conv3 = conv1x1(reduced_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, num_blocks, stride=1):
        downsample = None
        out_channels = channels * block.expansion

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # first block
        layers.append(block(self.in_channels, channels, stride, downsample))
        # rest blocks
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)

        return x


def load_pretrain_params(model, num_layers):
    pretrained_model = torchvision.models.__dict__[
        'resnet{}'.format(num_layers)](pretrained=True)

    model.conv1 = pretrained_model._modules['conv1']
    model.bn1 = pretrained_model._modules['bn1']
    model.relu = pretrained_model._modules['relu']
    model.maxpool = pretrained_model._modules['maxpool']

    model.layer1 = pretrained_model._modules['layer1']
    model.layer2 = pretrained_model._modules['layer2']
    model.layer3 = pretrained_model._modules['layer3']
    model.layer4 = pretrained_model._modules['layer4']

    del pretrained_model


def resnet18(pretrained=False):
    model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2])

    if pretrained is True:
        load_pretrain_params(model, 18)

    return model


def resnet50(pretrained=False):
    model = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3])

    if pretrained is True:
        load_pretrain_params(model, 50)

    return model
