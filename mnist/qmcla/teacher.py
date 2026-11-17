import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    def __init__(self, p):
        super(Dropout, self).__init__()

        self.p = p

    def forward(self, x):

        mask = torch.empty_like(x).bernoulli_(1 - self.p) / (1 - self.p)

        return x * mask


class ConvNet_Simple(nn.Module):
    def __init__(self, cfg):
        super(ConvNet_Simple, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)

        self.pool = nn.MaxPool2d(2, 2)  # halves spatial size

        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # after pooling twice: 28 -> 14 -> 7
        self.fc2 = nn.Linear(256, cfg["num_classes"])

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_candidates(self, indices):

        return None
