import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score
from torch.nn.utils import spectral_norm


# ============================================================
# Configuration
# ============================================================

cfg = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # 8 known classes
    "num_classes": 8,
    # feature dimension
    "feature_dim": 128,
    "VGG16": [
        32,
        64,
        "M",
        64,
        64,
        "M",
        128,
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        256,
        256,
        128,
        "M",
    ],
    "spectral_norm": True,
}


device = torch.device(cfg["device"])

print("Device:", device)


# ============================================================
# Known / unknown classes
# ============================================================

KNOWN_CLASSES = [0, 1, 4, 5, 6, 7, 8, 9]
UNKNOWN_CLASSES = [2, 3]


# ============================================================
# Dataset
# ============================================================


class FilteredCIFAR10(Dataset):
    def __init__(self, dataset, keep_classes, remap_labels=False):

        self.dataset = dataset
        self.remap_labels = remap_labels

        self.indices = [
            i for i, label in enumerate(dataset.targets) if int(label) in keep_classes
        ]

        self.mapping = {c: i for i, c in enumerate(sorted(keep_classes))}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        image, label = self.dataset[real_idx]

        label = int(label)

        if self.remap_labels:
            label = self.mapping[label]

        return image, label


# ============================================================
# CIFAR-10 transforms
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
        ),
    ]
)


test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
        ),
    ]
)


# ============================================================
# Load CIFAR-10
# ============================================================

train_raw = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)


test_raw = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)


# ============================================================
# Create datasets
# ============================================================

train_dataset = FilteredCIFAR10(train_raw, KNOWN_CLASSES, remap_labels=True)


known_test_dataset = FilteredCIFAR10(test_raw, KNOWN_CLASSES, remap_labels=True)


unknown_test_dataset = FilteredCIFAR10(test_raw, UNKNOWN_CLASSES, remap_labels=False)


print()
print("Training samples:", len(train_dataset))
print("Known test samples:", len(known_test_dataset))
print("Unknown test samples:", len(unknown_test_dataset))


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)


train_eval_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)


known_loader = DataLoader(
    known_test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)


unknown_loader = DataLoader(
    unknown_test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)


# ============================================================
# VGG
# ============================================================


class VGG(nn.Module):
    def __init__(self, cfg, spectral_normalization=True):

        super().__init__()

        self.features = self._make_layers(cfg["VGG16"])

        self.fc = nn.Linear(cfg["feature_dim"], cfg["feature_dim"])

        self.classifier = nn.Linear(cfg["feature_dim"], cfg["num_classes"])

        if spectral_normalization:
            self.apply_spectral_norm()

    def forward(self, x, return_features=False):

        x = self.features(x)

        x = torch.flatten(x, 1)

        z = self.fc(x)

        z = F.relu(z)

        logits = self.classifier(z)

        if return_features:
            return logits, z

        return logits

    def _make_layers(self, cfg):

        layers = []

        in_channels = 3

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

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

    def apply_spectral_norm(self):

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                spectral_norm(module)


# ============================================================
# Create model
# ============================================================

model = VGG(cfg, spectral_normalization=True).to(device)


print()
print(model)


# ============================================================
# Training
# ============================================================


def train(model, loader, device, epochs=30):

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()

        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y in loader:
            x = x.to(device, non_blocking=True)

            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = F.cross_entropy(logits, y)

            loss.backward()

            optimizer.step()

            loss_sum += loss.item() * x.size(0)

            prediction = logits.argmax(dim=1)

            correct += (prediction == y).sum().item()

            total += x.size(0)

        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}: "
            f"loss={loss_sum / total:.4f}, "
            f"acc={correct / total:.4f}"
        )


# ============================================================
# Train
# ============================================================

train(model, train_loader, device, epochs=30)


# ============================================================
# Feature extraction
# ============================================================


@torch.no_grad()
def extract_features(model, loader, device):

    model.eval()

    features = []
    labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        _, z = model(x, return_features=True)

        features.append(z.cpu())

        labels.append(y)

    return (torch.cat(features), torch.cat(labels))


print()
print("Extracting training features...")


train_features, train_labels = extract_features(model, train_eval_loader, device)


print("Feature shape:", train_features.shape)


# ============================================================
# GDA
# ============================================================


class GDA:
    def __init__(self, num_classes, eps=1e-3):

        self.num_classes = num_classes
        self.eps = eps

    def fit(self, features, labels):

        device = features.device

        n, d = features.shape

        # --------------------------------
        # Class means
        # --------------------------------

        means = []

        for c in range(self.num_classes):
            x_c = features[labels == c]

            mean_c = x_c.mean(dim=0)

            means.append(mean_c)

        self.means = torch.stack(means)

        # --------------------------------
        # Shared covariance
        # --------------------------------

        covariance = torch.zeros(d, d, device=device)

        count = 0

        for c in range(self.num_classes):
            x_c = features[labels == c]

            centered = x_c - self.means[c]

            covariance += centered.T @ centered

            count += x_c.shape[0]

        covariance /= count - self.num_classes

        # --------------------------------
        # Regularization
        # --------------------------------

        covariance += self.eps * torch.eye(d, device=device)

        self.covariance = covariance

        self.precision = torch.linalg.inv(covariance)

        _, self.log_det = torch.linalg.slogdet(covariance)

        # Equal class priors
        self.prior = torch.ones(self.num_classes, device=device) / self.num_classes

        print("GDA fitted.")

        print("Feature dimension:", d)

    def class_log_prob(self, z):

        diff = z[:, None, :] - self.means[None, :, :]

        mahal = torch.einsum("bcd,de,bce->bc", diff, self.precision, diff)

        d = z.shape[1]

        log_prob = -0.5 * mahal - 0.5 * self.log_det - 0.5 * d * np.log(2 * np.pi)

        return log_prob

    def log_prob(self, z):

        class_log_prob = self.class_log_prob(z)

        return torch.logsumexp(class_log_prob + torch.log(self.prior)[None, :], dim=1)


# ============================================================
# Fit GDA
# ============================================================

train_features = train_features.to(device)
train_labels = train_labels.to(device)


gda = GDA(num_classes=8, eps=1e-3)


gda.fit(train_features, train_labels)


# ============================================================
# Uncertainty
# ============================================================


@torch.no_grad()
def evaluate_uncertainty(model, gda, loader, device):

    model.eval()

    epistemic = []
    aleatoric = []

    predictions = []
    labels = []

    log_density = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        # --------------------------------
        # Neural network
        # --------------------------------

        logits, z = model(x, return_features=True)

        # --------------------------------
        # Predictive probability
        # --------------------------------

        probabilities = F.softmax(logits, dim=1)

        # --------------------------------
        # Aleatoric uncertainty
        #
        # Predictive entropy
        # --------------------------------

        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)

        # --------------------------------
        # GDA density
        # --------------------------------

        density = gda.log_prob(z)

        # DDU-style epistemic score
        #
        # lower density = higher score
        # --------------------------------

        epistemic_score = -density

        epistemic.append(epistemic_score.cpu())

        aleatoric.append(entropy.cpu())

        log_density.append(density.cpu())

        predictions.append(probabilities.argmax(dim=1).cpu())

        labels.append(y)

    return {
        "epistemic": torch.cat(epistemic),
        "aleatoric": torch.cat(aleatoric),
        "log_density": torch.cat(log_density),
        "prediction": torch.cat(predictions),
        "labels": torch.cat(labels),
    }


# ============================================================
# Evaluate known samples
# ============================================================

print()
print("Evaluating known classes...")

known = evaluate_uncertainty(model, gda, known_loader, device)


# ============================================================
# Evaluate unknown samples
# ============================================================

print("Evaluating unknown classes...")

unknown = evaluate_uncertainty(model, gda, unknown_loader, device)


# ============================================================
# Known accuracy
# ============================================================

known_accuracy = (known["prediction"] == known["labels"]).float().mean()


# ============================================================
# Print results
# ============================================================

print()
print("========== Known classes ==========")

print("Classes:", KNOWN_CLASSES)

print("Accuracy:", known_accuracy.item())

print("Epistemic mean:", known["epistemic"].mean().item())

print("Epistemic std:", known["epistemic"].std().item())

print("Aleatoric mean:", known["aleatoric"].mean().item())

print("Aleatoric std:", known["aleatoric"].std().item())

print("Log density mean:", known["log_density"].mean().item())


print()
print("========== Unknown classes ==========")

print("Classes:", UNKNOWN_CLASSES)

print("Epistemic mean:", unknown["epistemic"].mean().item())

print("Epistemic std:", unknown["epistemic"].std().item())

print("Aleatoric mean:", unknown["aleatoric"].mean().item())

print("Aleatoric std:", unknown["aleatoric"].std().item())

print("Log density mean:", unknown["log_density"].mean().item())


# ============================================================
# OOD AUROC
# ============================================================

scores = torch.cat([known["epistemic"], unknown["epistemic"]]).numpy()


ood_labels = np.concatenate(
    [np.zeros(len(known["epistemic"])), np.ones(len(unknown["epistemic"]))]
)


auroc = roc_auc_score(ood_labels, scores)


print()
print("OOD AUROC:", auroc)


# ============================================================
# Histograms
# ============================================================

known_epistemic = known["epistemic"].numpy()

unknown_epistemic = unknown["epistemic"].numpy()

known_aleatoric = known["aleatoric"].numpy()

unknown_aleatoric = unknown["aleatoric"].numpy()


# ============================================================
# Epistemic histogram
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    known_epistemic, bins=100, alpha=0.6, density=True, label="Known (0,1,4,5,6,7,8,9)"
)

plt.hist(unknown_epistemic, bins=100, alpha=0.6, density=True, label="Unknown (2,3)")

plt.xlabel("Epistemic uncertainty (-log density)")

plt.ylabel("Density")

plt.title("Epistemic Uncertainty")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# Aleatoric histogram
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    known_aleatoric, bins=100, alpha=0.6, density=True, label="Known (0,1,4,5,6,7,8,9)"
)

plt.hist(unknown_aleatoric, bins=100, alpha=0.6, density=True, label="Unknown (2,3)")

plt.xlabel("Aleatoric uncertainty (entropy)")

plt.ylabel("Density")

plt.title("Aleatoric Uncertainty")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# Aleatoric vs Epistemic
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(known_aleatoric, known_epistemic, s=4, alpha=0.3, label="Known")

plt.scatter(unknown_aleatoric, unknown_epistemic, s=4, alpha=0.3, label="Unknown")

plt.xlabel("Aleatoric uncertainty")

plt.ylabel("Epistemic uncertainty")

plt.title("Aleatoric vs Epistemic Uncertainty")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()
