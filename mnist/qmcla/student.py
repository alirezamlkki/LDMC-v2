import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Dropout):
    def __init__(self, p, m_shape, h_ch, LCMC=True, is_2D=False):
        super(Attention, self).__init__()

        mid_h = 8

        self.agg1 = nn.Linear(m_shape, mid_h)
        self.agg2 = nn.Linear(mid_h, h_ch)
        self.p = p
        self.LCMC = LCMC
        self.is_2D = is_2D

    def forward(self, x, Q=None):

        if not (self.LCMC):
            return x

        if self.LCMC == "rand":
            Q = torch.rand_like(Q)

        Q = sum(Q) / len(Q)

        gamma_i = F.relu(self.agg1(Q))  # (batch, hidden_dim) each
        gamma = self.agg2(gamma_i)  # (batch, hidden_dim) each

        if self.is_2D:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        return F.sigmoid(gamma) * x


class ConvNet_s(nn.Dropout):
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

        self.soft_drop1 = Attention(cfg["p"], 32 + 9, 32, cfg["LDMC"], True)

        self.soft_drop2 = Attention(cfg["p"], 352, 64, cfg["LDMC"], True)

        self.fc1_ = nn.Linear(64 * 7 * 7, 256)
        self.soft_drop3 = Attention(cfg["p"], 3392, 256, cfg["LDMC"])
        self.soft_drop4 = Attention(cfg["p"], 3392, 64, cfg["LDMC"])

        self.fc2_ = nn.Linear(256, cfg["num_classes"])

        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 4)
        self.fc5 = nn.Linear(12, 12)
        self.fc6 = nn.Linear(12, 12)

        self.fc7 = nn.Linear(12, 1)

    def forward(self, x, masks):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # for m in masks:

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # for m in masks:

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1_(x))
        stack_masks = []
        for m in masks:
            mask = torch.cat([m[0][0], m[0][1]], dim=0).unsqueeze(dim=0)
            stack_masks.append(mask)
        x = self.soft_drop3(x, mask)

        cls = self.fc2_(x)

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        x = F.silu(torch.cat((x, cls), dim=-1))

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        unc = F.softplus(self.fc7(x))

        return cls, unc
