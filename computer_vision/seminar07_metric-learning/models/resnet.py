#-*- coding: utf8 -*-
import torch.nn as nn
import math
from .scoring import ArcFaceScoring


__all__ = ['ResNet', 'resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class BasicBlockV3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV3, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = nn.PReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.activation = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class HeadE(nn.Module):

    def __init__(self, input_size, emb_size):
        "FC layer, returns embedding"
        super(HeadE, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.drop = nn.Dropout(0.4, inplace=True)
        self.fc = nn.Linear(input_size, emb_size)
        self.bn2 = nn.BatchNorm1d(emb_size, affine=False)

    def forward(self, input):
        input = input.view(input.size(0), -1)

        x = self.bn1(input)
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, scoring, layers, emb_size=1000, preserve_resolution=True, head_size=7*7*512):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if preserve_resolution:
            self.body = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU()
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.layer1 = self._make_layer(block,  64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.head = HeadE(head_size, emb_size)
        self.scoring = scoring

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.criterion = None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, target=None, calc_scores=True):
        x = self.body(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        embeddings = self.head(x)

        scores = None

        if calc_scores:
            scores = self.scoring(embeddings, target)

            if self.criterion is not None:
                return embeddings, None, self.criterion(scores, target)

        return embeddings, scores, None


def resnet(depth, num_classes, m=0.5, s=64., scoring=None, **kwargs):
    if depth >= 101:
        block = Bottleneck
    else:
        block = BasicBlockV3

    if depth == 18:
        units = [2, 2, 2, 2]
    elif depth == 34:
        units = [3, 4, 6, 3]
    elif depth == 49:
        units = [3, 4, 14, 3]
    elif depth == 50:
        units = [3, 4, 14, 3]
    elif depth == 74:
        units = [3, 6, 24, 3]
    elif depth == 90:
        units = [3, 8, 30, 3]
    elif depth == 100:
        units = [3, 13, 30, 3]
    elif depth == 101:
        units = [3, 4, 23, 3]
    elif depth == 152:
        units = [3, 8, 36, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(depth))

    emb_size = kwargs.get('emb_size')

    if scoring is None:
        # default
        scoring = ArcFaceScoring(m, s, emb_size, num_classes)

    model = ResNet(block, scoring, units, **kwargs)

    return model

def create(opts):
    return resnet(100, opts['classes'], emb_size=opts['emb_size'], head_size=opts['head_size']).cuda()