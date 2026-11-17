import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Dropout):
    def __init__(self, p, m_shape, h_ch, LCMC=True, is_2D=False):
        super(Attention, self).__init__()

        mid_h = 64

        self.agg1 = nn.Linear(m_shape, mid_h)
        self.agg2 = nn.Linear(mid_h, h_ch)
        self.p = p
        self.LCMC = LCMC
        self.is_2D = is_2D

    def forward(self, x, Q=None):

        if not (self.LCMC):
            return x

        elif self.LCMC == "rand":
            Q = torch.rand_like(Q[0])
        else:
            Q = sum(Q) / len(Q)

        gamma_i = F.relu(self.agg1(Q))  # (batch, hidden_dim) each
        gamma = self.agg2(gamma_i)  # (batch, hidden_dim) each

        if self.is_2D:
            gamma = gamma[:, None, None]

        return F.sigmoid(gamma) * x


class VGG(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.LDMC = cfg["LDMC"]
        self.p = cfg["p"]
        num_classes = cfg["num_classes"]

        last_f_num = 128

        self.features = self._make_layers(cfg)

        self.dp_1 = Attention(
            self.p, last_f_num * 2, last_f_num, LCMC=cfg["LDMC"], is_2D=False
        )
        self.fc2 = nn.Linear(last_f_num, last_f_num)
        self.classifier = nn.Linear(last_f_num, num_classes)
        self.fc3 = nn.Linear(last_f_num, last_f_num // 2)
        self.dp_2 = Attention(
            self.p, last_f_num * 2, last_f_num // 2, LCMC=cfg["LDMC"], is_2D=False
        )

        self.fc4 = nn.Linear(last_f_num // 2, 4)

        self.fc5 = nn.Linear(12, 12)
        self.fc6 = nn.Linear(12, 12)

        self.fc7 = nn.Linear(12, 1)

    def forward(self, x, Q):

        c_id = 0
        for layer in self.features:
            if isinstance(layer, Attention):
                stack_masks = []
                for m in Q:
                    mask = torch.cat([m[c_id][0], m[c_id][1]], dim=0)
                    stack_masks.append(mask)
                x = layer(x, stack_masks)

                c_id += 2
            else:
                x = layer(x)

        x = torch.flatten(x, 1)

        stack_masks = []
        for m in Q:
            mask = torch.cat([m[c_id][0], m[c_id][1]], dim=0)
            stack_masks.append(mask)

        x = self.dp_1(x, stack_masks)
        x = F.relu(self.fc2(x))
        cls = self.classifier(x)

        x = F.relu(self.fc3(x))

        x = self.dp_2(x, stack_masks)
        x = self.fc4(x)

        x = F.silu(torch.cat((x, cls), dim=-1))

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        unc = F.softplus(self.fc7(x))

        return cls, unc

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        h, w = 32, 32  # CIFAR-10 input size
        cfg = config["VGG16"]

        for i, v in enumerate(cfg):
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                h //= 2
                w //= 2

            elif v == "P":
                print(cfg[i - 2] * 9, cfg[i - 1])
                layers.append(
                    Attention(
                        config["p"],
                        cfg[i - 2] * 9 + cfg[i - 1],
                        in_channels,
                        config["LDMC"],
                        is_2D=True,
                    )
                )

            else:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = v

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)


def vgg16_cifar_student(cfgs):
    return VGG(cfgs)
