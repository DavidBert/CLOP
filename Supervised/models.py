import torch
from torch import nn
import numpy as np
from layers import CLOPLayer
import torchvision.models as models


class MNISTClassifier(nn.Module):
    def __init__(self, img_size=(1, 32, 32), regul=None, p=0.9):
        super(MNISTClassifier, self).__init__()

        self.regul = regul

        modules = []
        modules.append(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        )
        modules.append(nn.ReLU(True))
        if self.regul == "batch_norm":
            modules.append(nn.BatchNorm2d(32))
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        )
        modules.append(nn.ReLU(True))
        if self.regul == "batch_norm":
            modules.append(nn.BatchNorm2d(64))
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        )
        modules.append(nn.ReLU(True))
        if self.regul == "batch_norm":
            modules.append(nn.BatchNorm2d(128))
        if self.regul == "clop":
            modules.append(CLOPLayer(p))
        if self.regul == "dropout":
            modules.append(nn.Dropout2d(p))

        self.conv_feat = nn.Sequential(*modules)

        self.conv_feat_size = self.conv_feat(torch.zeros(1, *img_size)).shape[1:]
        self.dense_feature_size = np.prod(self.conv_feat_size)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.dense_feature_size, out_features=512),
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(True),
            nn.Linear(in_features=100, out_features=10),
        )

    def forward(self, x):
        x = self.conv_feat(x)
        x = x.view(-1, self.dense_feature_size)
        x = self.classifier(x)
        y = torch.log_softmax(x, 1)
        return y


class VGG11(nn.Module):
    def __init__(self, regul=None, p=0.7):
        super(VGG11, self).__init__()
        self.regul = regul
        if self.regul == "batch_norm":
            vgg = models.vgg11_bn(pretrained=False)
        else:
            vgg = models.vgg11(pretrained=False)
        self.conv_feat = vgg.features
        if self.regul == "clop":
            self.conv_feat.add_module("clop", CLOPLayer(p))
        if self.regul == "dropout":
            self.conv_feat.add_module("dropout", nn.Dropout2d(p))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=10, bias=True),
        )
        self.dropout = nn.Dropout2d(p)
        self.clop = CLOPLayer(p)
        self.batchnorm = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv_feat(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        y = torch.log_softmax(x, 1)
        return y


class VGG9(nn.Module):
    def __init__(self, regul=None, p=0.7):
        super(VGG9, self).__init__()
        self.regul = regul
        if self.regul == "batch_norm":
            vgg = models.vgg11_bn(pretrained=False)
        else:
            vgg = models.vgg11(pretrained=False)

        self.conv_feat = vgg.features[:-5]
        if self.regul == "clop":
            self.conv_feat.add_module("clop", CLOPLayer(p))
        if self.regul == "dropout":
            self.conv_feat.add_module("dropout", nn.Dropout2d(p))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=18432, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=10, bias=True),
        )

    def forward(self, x):
        x = self.conv_feat(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        y = torch.log_softmax(x, 1)
        return y
