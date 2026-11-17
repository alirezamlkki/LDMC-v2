import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlledDropout(nn.Module):
    def __init__(self, p, shape, device, M=10, rw=0.5):
        super(ControlledDropout, self).__init__()
        self.store = []
        for i in range(M):
            mask = torch.empty(shape, device=device).bernoulli_(1 - p)
            self.store.append(mask)

        self.rw = rw
        self.p = p

    def forward(self, x, index):
        selected = torch.stack([self.store[i] for i in index], dim=0)
        random_walk = torch.ones_like(selected, device=x.device) * self.rw
        random_walk = random_walk + selected * (2 * self.p - 1) / (1 - self.p) * self.rw
        random_walk = torch.bernoulli(random_walk)

        mask = (selected.int() ^ random_walk.int()) / (1 - self.p)

        return x * mask

    def get_candidates(self, indices):
        return [self.store[i] for i in indices]


class ConvNet(nn.Module):
    def __init__(self, cfg):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.drop1 = ControlledDropout(
            cfg["p"], [32, 14, 14], cfg["device"], M=cfg["M"], rw=cfg["rw"]
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.drop2 = ControlledDropout(
            cfg["p"], [64, 7, 7], cfg["device"], M=cfg["M"], rw=cfg["rw"]
        )

        self.pool = nn.MaxPool2d(2, 2)  # halves spatial size

        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # after pooling twice: 28 -> 14 -> 7
        self.fc2 = nn.Linear(256, cfg["num_classes"])

    def forward(self, x, index):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x, index)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop2(x, index)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_candidates(self, indices):
        mask1 = self.drop1.get_candidates(indices)
        mask2 = self.drop2.get_candidates(indices)

        return [mask1, mask2]
