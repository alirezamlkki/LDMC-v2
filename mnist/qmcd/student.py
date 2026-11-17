import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDropout(nn.Module):
    def __init__(self, p, C, LCMC=True):
        super(SoftDropout, self).__init__()

        self.agg = nn.Conv3d(3, 1, kernel_size=1, bias=True)
        self.p = p
        self.LCMC = LCMC

    def forward(self, x, masks):

        if not self.LCMC:
            return x

        masks = torch.stack(masks, dim=0).unsqueeze(0)
        mask = self.agg(masks)
        mask = F.sigmoid(mask.squeeze(dim=1)) / (1 - self.p)

        return x * mask


class ConvNet_s(nn.Module):
    def __init__(self, cfg):
        super(ConvNet_s, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1, bias=False
        )  # input: (1, 28, 28), output: (32, 28, 28)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1, bias=False
        )  # input: (32,32,32), output: (64,32,32)
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial size

        self.soft_drop1 = SoftDropout(cfg["p"], cfg["LDMC"])
        self.soft_drop2 = SoftDropout(cfg["p"], cfg["LDMC"])

        self.fc1_ = nn.Linear(64 * 7 * 7, 256)
        self.fc2_ = nn.Linear(256, cfg["num_classes"])

        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 4)
        self.fc5 = nn.Linear(12, 12)
        self.fc6 = nn.Linear(12, 12)

        self.fc7 = nn.Linear(12, 1)

        self.p = cfg["p"]

    def forward(self, x, masks):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.soft_drop1(x, masks[0])

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.soft_drop1(x, masks[1])

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1_(x))
        cls = self.fc2_(x)

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        x = F.silu(torch.cat((x, cls), dim=-1))

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        unc = F.softplus(self.fc7(x))

        return cls, unc
