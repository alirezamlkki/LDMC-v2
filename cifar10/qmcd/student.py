import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDropout(nn.Module):
    def __init__(self, p, LCMC=True):
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


class VGG(nn.Module):
    def __init__(self, cfg, num_classes=8):
        super().__init__()

        self.features = self._make_layers(cfg)

        self.fc2 = nn.Linear(256, num_classes)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 4)
        self.fc5 = nn.Linear(12, 12)
        self.fc6 = nn.Linear(12, 12)

        self.fc7 = nn.Linear(12, 1)

        self.LDMC = cfg["LDMC"]
        self.p = cfg["p"]

    def forward(self, x, masks=None):
        c_mask = 0
        for layer in self.features:
            if isinstance(layer, SoftDropout):
                new_mask = [
                    torch.bernoulli(torch.full(m.shape, self.p)).to("cuda")
                    for m in masks[c_mask]
                ]
                x = layer(x, new_mask)
                c_mask += 1
            else:
                x = layer(x)

        x = torch.flatten(x, 1)

        cls = self.fc2(x)

        x = F.relu(self.fc3(x))

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

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                h //= 2
                w //= 2

            elif v == "P":
                layers.append(SoftDropout(config["p"], config["LDMC"]))

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
