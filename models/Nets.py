#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models  # 导入 torchvision 模型
from torchvision.models import resnet18, vgg16

class MobileNetCifar(nn.Module):
    def __init__(self, args):
        super(MobileNetCifar, self).__init__()
        # 使用 torchvision 的 mobilenet_v2
        self.model = models.mobilenet_v2(num_classes=args.num_classes)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# ============================ 新 ============================
class VGG11(nn.Module):  # VGG11 适配 CIFAR10 (3通道, 32x32)
    def __init__(self, args):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),  # CIFAR10 -> 128*8*8
            nn.ReLU(True),
            nn.Linear(256, args.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG16(nn.Module): # ImageNet (224×224×3)
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.model = vgg16(num_classes=args.num_classes)

    ''' 修改第一层卷积以适配 CIFAR (32x32, 3 通道)
        self.model.features[0] = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=3,  # 原来是 3x3
            stride=1,       # CIFAR 不能下采样太快
            padding=1
        )'''
    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module): # ResNet-18（简化版） ImageNet 数据集（224×224 像素）
    def __init__(self, args):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=args.num_classes)
        '''若要适配CIFAR数据集（32×32像素），将第一个卷积层从 7x7 改为 3x3，步长从 2 改为 1，padding 从 3 改为 1
        self.model.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=3,  # 原 7
            stride=1,       # 原 2
            padding=1,      # 原 3
            bias=False
        )'''
    def forward(self, x):
        return self.model(x)

class LR(nn.Module): # 逻辑回归 MNIST
    def __init__(self, dim_in, dim_out):
        super(LR, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        return self.fc(x)

class MobileNetCifar(nn.Module): # ImageNet (224×224×3)
    def __init__(self, args):
        super(MobileNetCifar, self).__init__()
        # 使用 torchvision 的 mobilenet_v2
        self.model = models.mobilenet_v2(num_classes=args.num_classes)
        ''' 若CIFAR:修改第一层卷积: kernel_size=3, stride=1, padding=1，避免过快下采样
        self.model.features[0][0] = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )'''
    def forward(self, x):
        return self.model(x)

class LeNet5(nn.Module): # LeNet5（轻量级 CNN） MNIST
    def __init__(self, args):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, 5)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

