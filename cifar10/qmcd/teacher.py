import torch
import torch.nn as nn


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


class VGG(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = cfg["device"]
        self.p = cfg["p"]
        self.M = cfg["M"]
        self.rw = cfg["rw"]

        self.features = self._make_layers(cfg["VGG16"])
        self.classifier = nn.Linear(256, cfg["num_classes"])

    def forward(self, x, indices):

        for layer in self.features:
            if isinstance(layer, ControlledDropout):
                x = layer(x, indices)
            else:
                x = layer(x)

        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        h, w = 32, 32  # CIFAR-10 input size

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                h //= 2
                w //= 2

            elif v == "P":
                shape = (in_channels, h, w)
                layers.append(
                    ControlledDropout(
                        p=self.p, shape=shape, device=self.device, M=self.M, rw=self.rw
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

    def get_candidates(self, indices):
        masks = []
        for layer in self.features:
            if isinstance(layer, ControlledDropout):
                masks.append(layer.get_candidates(indices))

        return masks


def vgg16_cifar_teacher(config):
    return VGG(config)
