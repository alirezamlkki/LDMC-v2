import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class VGG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg["device"]
        self.p = cfg["p"]
        self.M = cfg["M"]
        self.features = self._make_layers(cfg["VGG16"])
        self.fc = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, cfg["num_classes"])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        return self.classifier(x)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        h, w = 32, 32  # CIFAR-10 input size
        for i, v in enumerate(cfg):
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                h //= 2
                w //= 2
            elif v == "P":
                shape = (in_channels, h, w)
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

    @torch.no_grad()
    def fuse_bn(self):
        """
        Returns a NEW model (deep copy) where every Conv2d -> BatchNorm2d pair
        in `features` is replaced by a single fused Conv2d (bias=True).
        Must be called AFTER training, with the model in eval() mode so that
        BatchNorm's running_mean/running_var are the ones actually used
        for fusion (they must match what eval-mode inference already uses).
        """
        assert not self.training, (
            "Call model.eval() before fuse_bn() -- fusion uses running stats."
        )

        fused_model = copy.deepcopy(self)
        old_layers = list(self.features.children())
        new_layers = []

        i = 0
        while i < len(old_layers):
            layer = old_layers[i]
            # Detect Conv2d immediately followed by BatchNorm2d
            if (
                isinstance(layer, nn.Conv2d)
                and i + 1 < len(old_layers)
                and isinstance(old_layers[i + 1], nn.BatchNorm2d)
            ):
                conv = layer
                bn = old_layers[i + 1]
                fused_conv = self._fuse_conv_bn(conv, bn)
                new_layers.append(fused_conv)
                i += 2  # skip both conv and bn, they're now one layer
            else:
                new_layers.append(copy.deepcopy(layer))
                i += 1

        fused_model.features = nn.Sequential(*new_layers)
        fused_model.eval()
        return fused_model

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        """
        Computes the fused Conv2d equivalent to Conv2d(bias=False) -> BatchNorm2d.

        W_fused = W_conv * gamma / sqrt(running_var + eps)
        b_fused = beta - gamma * running_mean / sqrt(running_var + eps)
        """
        fused_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).to(conv.weight.device)

        gamma = bn.weight
        beta = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(running_var + eps)  # (out_channels,)
        scale = gamma / std  # (out_channels,)

        # W_conv shape: (out_channels, in_channels, kH, kW) -> scale per out_channel
        fused_conv.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        fused_conv.bias.copy_(beta - running_mean * scale)

        return fused_conv


def vgg16_cifar_teacher(config):
    return VGG(config)
