import torch
import torch.nn as nn
import os


class LeNet5(nn.Module):
    def __init__(self, out_features):
        super(LeNet5, self).__init__()
        # Input(1, 32, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.LocalResponseNorm(5)
        )
        # Shape(6, 28, 28)
        self.pool1 = nn.AvgPool2d(2)
        # Shape(6, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.Tanh()
        )
        # Shape(16, 10, 10)
        self.pool2 = nn.AvgPool2d(2)
        # Shape(16, 5, 5)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, 5),
            nn.Tanh()
        )
        # Shape(120)
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh()
        )
        # Shape(84)
        self.fc2 = nn.Sequential(
            nn.Linear(84, 10),
            nn.Tanh()
        )
        # Output(10)
        self.out = nn.Linear(10, out_features)



    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


def save_model(net):
    """
    动态保存模型
    :param net:
    :return:
    """
    i = 0.8
    while True:
        # 如果模型存在了，继续下去，否则直接存
        if os.access("./resources/models/model@42app_time{}.pth".format(i), os.F_OK):
            i = float(format(i + 0.1, ".1f"))
            continue
        torch.save(net, "./resources/models/model@42app_time{}.pth".format(i))
        break