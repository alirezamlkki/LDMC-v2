import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score

import ddu_dirty_mnist
import matplotlib.pyplot as plt
import pickle
import numpy as np


# ============================================================
# Dataset
# ============================================================


class FilteredMNIST(Dataset):
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

        image = self.dataset.data[real_idx]
        label = int(self.dataset.targets[real_idx])

        image = image.float() / 255.0
        image = image.unsqueeze(0)

        if self.remap_labels:
            label = self.mapping[label]

        return image, label


# ============================================================
# LeNet
# ============================================================

import torch.nn.utils as utils


class LeNet(nn.Module):
    def __init__(self, feature_dim=128, num_classes=8):

        super().__init__()

        self.features = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(1, 6, kernel_size=5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            utils.spectral_norm(nn.Conv2d(6, 16, kernel_size=5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            utils.spectral_norm(nn.Linear(16 * 4 * 4, feature_dim)), nn.ReLU()
        )

        self.classifier = utils.spectral_norm(nn.Linear(feature_dim, num_classes))

    def forward(self, x, return_features=False):

        x = self.features(x)

        x = torch.flatten(x, 1)

        z = self.fc(x)

        logits = self.classifier(z)

        if return_features:
            return logits, z

        return logits


# ============================================================
# Training
# ============================================================


def train(model, loader, device, epochs=10):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(epochs):
        total = 0
        correct = 0
        loss_sum = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = F.cross_entropy(logits, y)

            loss.backward()

            optimizer.step()

            loss_sum += loss.item() * x.size(0)

            pred = logits.argmax(1)

            correct += (pred == y).sum().item()

            total += x.size(0)

        print(
            f"Epoch {epoch + 1}: loss={loss_sum / total:.4f}, acc={correct / total:.4f}"
        )


# ============================================================
# Feature extraction
# ============================================================


@torch.no_grad()
def extract_features(model, loader, device):

    model.eval()

    features = []
    labels = []

    for x, y in loader:
        x = x.to(device)

        _, z = model(x, return_features=True)

        features.append(z.cpu())

        labels.append(y)

    return (torch.cat(features), torch.cat(labels))


# ============================================================
# Gaussian Discriminant Analysis
# ============================================================


class GDA:
    def __init__(self, num_classes, eps=1e-5):

        self.num_classes = num_classes
        self.eps = eps

    def fit(self, features, labels):

        device = features.device

        n, d = features.shape

        means = []

        for c in range(self.num_classes):
            x_c = features[labels == c]

            means.append(x_c.mean(0))

        self.means = torch.stack(means)

        covariance = torch.zeros(d, d, device=device)

        count = 0

        for c in range(self.num_classes):
            x_c = features[labels == c]

            centered = x_c - self.means[c]

            covariance += centered.T @ centered

            count += len(x_c)

        covariance /= count - self.num_classes

        covariance += self.eps * torch.eye(d, device=device)

        self.precision = torch.linalg.inv(covariance)

        _, self.log_det = torch.linalg.slogdet(covariance)

        self.prior = torch.tensor(
            [(labels == c).float().mean() for c in range(self.num_classes)],
            device=device,
        )

        print("GDA fitted")

    def log_prob(self, z):

        diff = z[:, None, :] - self.means[None, :, :]

        mahal = torch.einsum("bcd,de,bce->bc", diff, self.precision, diff)

        d = z.shape[1]

        log_p = -0.5 * mahal - 0.5 * self.log_det - 0.5 * d * np.log(2 * np.pi)

        log_prior = torch.log(self.prior)

        return torch.logsumexp(log_p + log_prior, dim=1)


# ============================================================
# DDU uncertainty
# ============================================================


@torch.no_grad()
def uncertainty(model, gda, loader, device):

    model.eval()

    epistemic = []
    aleatoric = []

    prediction = []
    labels = []

    for x, y in loader:
        x = x.to(device)

        logits, z = model(x, return_features=True)

        prob = F.softmax(logits, dim=1)

        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)

        density = gda.log_prob(z)

        epistemic.append(-density.cpu())

        aleatoric.append(entropy.cpu())

        prediction.append(prob.argmax(1).cpu())

        labels.append(y)

    return {
        "epistemic": torch.cat(epistemic),
        "aleatoric": torch.cat(aleatoric),
        "prediction": torch.cat(prediction),
        "labels": torch.cat(labels),
    }


# ============================================================
# Main
# ============================================================


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    transform = transforms.ToTensor()

    # -----------------------------
    # Load MNIST
    # -----------------------------

    train_raw = datasets.MNIST("./data", train=True, download=True)

    test_raw = datasets.MNIST("./data", train=False, download=True)

    # -----------------------------
    # Known classes: 1-8
    # -----------------------------

    train_dataset = FilteredMNIST(
        train_raw, [1, 2, 3, 4, 5, 6, 7, 8], remap_labels=True
    )

    known_test = FilteredMNIST(test_raw, [1, 2, 3, 4, 5, 6, 7, 8], remap_labels=True)

    # -----------------------------
    # OOD: 0 and 9
    # -----------------------------

    unknown_test = FilteredMNIST(test_raw, [0, 9], remap_labels=False)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    train_eval_loader = DataLoader(train_dataset, batch_size=512)

    known_loader = DataLoader(known_test, batch_size=512)

    unknown_loader = DataLoader(unknown_test, batch_size=512)

    # -----------------------------
    # Train model
    # -----------------------------

    model = LeNet(feature_dim=128, num_classes=8).to(device)

    train(model, train_loader, device, epochs=30)

    # -----------------------------
    # Fit GDA
    # -----------------------------

    features, labels = extract_features(model, train_eval_loader, device)

    features = features.to(device)
    labels = labels.to(device)

    gda = GDA(num_classes=8)

    gda.fit(features, labels)

    # -----------------------------
    # Evaluate uncertainty
    # -----------------------------

    known = uncertainty(model, gda, known_loader, device)

    unknown = uncertainty(model, gda, unknown_loader, device)

    # ============================================================
    # Convert tensors to numpy
    # ============================================================

    known_epistemic = known["epistemic"].numpy()
    unknown_epistemic = unknown["epistemic"].numpy()

    known_aleatoric = known["aleatoric"].numpy()
    unknown_aleatoric = unknown["aleatoric"].numpy()

    with open("amb-data.pkl", "wb") as f:
        pickle.dump(
            {
                "known_epistemic": known_epistemic,
                "unknown_epistemic": unknown_epistemic,
                "known_aleatoric": known_aleatoric,
                "unknown_aleatoric": unknown_aleatoric,
            },
            f,
        )

    # ============================================================
    # Plot epistemic uncertainty
    # ============================================================

    plt.figure(figsize=(8, 5))

    max_value = 4000
    known_epistemic = known_epistemic[known_epistemic <= max_value]
    unknown_epistemic = unknown_epistemic[unknown_epistemic <= max_value]

    plt.hist(
        known_epistemic, bins=100, alpha=0.6, density=True, label="Known (digits 1-8)"
    )

    plt.hist(
        unknown_epistemic,
        bins=100,
        alpha=0.6,
        density=True,
        label="Unknown (digits 0,9)",
    )

    plt.xlabel("Epistemic uncertainty (-log density)")
    plt.ylabel("Density")
    plt.title("Epistemic uncertainty distribution")
    plt.legend()
    plt.grid()

    plt.savefig("res_ep.png")

    # ============================================================
    # Plot aleatoric uncertainty
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.hist(
        known_aleatoric, bins=100, alpha=0.6, density=True, label="Known (digits 1-8)"
    )

    plt.hist(
        unknown_aleatoric,
        bins=100,
        alpha=0.6,
        density=True,
        label="Unknown (digits 0,9)",
    )

    plt.xlabel("Aleatoric uncertainty (entropy)")
    plt.ylabel("Density")
    plt.title("Aleatoric uncertainty distribution")
    plt.legend()
    plt.grid()

    plt.savefig("res_al.png")

    print("\n========== Known (1-8) ==========")

    print("Accuracy:", (known["prediction"] == known["labels"]).float().mean().item())

    print("Epistemic mean:", known["epistemic"].mean().item())

    print("Epistemic std:", known["epistemic"].std().item())

    print("Aleatoric mean:", known["aleatoric"].mean().item())

    print("\n========== Unknown (0,9) ==========")

    print("Epistemic mean:", unknown["epistemic"].mean().item())

    print("Epistemic std:", unknown["epistemic"].std().item())

    print("Aleatoric mean:", unknown["aleatoric"].mean().item())

    # -----------------------------
    # AUROC
    # -----------------------------

    scores = torch.cat([known["epistemic"], unknown["epistemic"]]).numpy()

    labels = np.concatenate(
        [np.zeros(len(known["epistemic"])), np.ones(len(unknown["epistemic"]))]
    )

    auroc = roc_auc_score(labels, scores)

    print("\nOOD AUROC:", auroc)

    # ============================================================
    # Ambiguous MNIST test loader
    # ============================================================

    test_ambiguous = ddu_dirty_mnist.AmbiguousMNIST(
        "./data", train=False, download=True, device="cuda"
    )

    ambiguous_loader = DataLoader(test_ambiguous, batch_size=512, shuffle=False)

    # ============================================================
    # Evaluate uncertainty
    # ============================================================

    ambiguous = uncertainty(model, gda, ambiguous_loader, device)

    # ============================================================
    # Separate Known / Unknown
    # ============================================================

    amb_labels = ambiguous["labels"].cpu()

    # Known = 1-8
    known_mask = (amb_labels >= 1) & (amb_labels <= 8)

    # Unknown = 0,9
    unknown_mask = (amb_labels == 0) | (amb_labels == 9)

    amb_known_epistemic = ambiguous["epistemic"][known_mask].numpy()

    amb_unknown_epistemic = ambiguous["epistemic"][unknown_mask].numpy()

    amb_known_aleatoric = ambiguous["aleatoric"][known_mask].numpy()

    amb_unknown_aleatoric = ambiguous["aleatoric"][unknown_mask].numpy()

    # ============================================================
    # Epistemic AUROC
    # ============================================================

    epistemic_scores = ambiguous["epistemic"].cpu().numpy()

    auroc_epistemic = roc_auc_score(known_mask, epistemic_scores)

    # ============================================================
    # Print statistics
    # ============================================================

    print("\n========== Ambiguous MNIST: Known ==========")

    print("Number of samples:", len(amb_known_epistemic))

    print("Epistemic mean:", amb_known_epistemic.mean())

    print("Epistemic std:", amb_known_epistemic.std())

    print("Aleatoric mean:", amb_known_aleatoric.mean())

    print("\n========== Ambiguous MNIST: Unknown ==========")

    print("Number of samples:", len(amb_unknown_epistemic))

    print("Epistemic mean:", amb_unknown_epistemic.mean())

    print("Epistemic std:", amb_unknown_epistemic.std())

    print("Aleatoric mean:", amb_unknown_aleatoric.mean())

    print("AUROC:", auroc_epistemic)

    # ============================================================
    # Epistemic histogram
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.hist(amb_known_epistemic, bins=100, density=True, alpha=0.6, label="Known")

    plt.hist(amb_unknown_epistemic, bins=100, density=True, alpha=0.6, label="Unknown")

    plt.xlabel("Epistemic uncertainty (-log density)")

    plt.ylabel("Density")

    plt.title("Ambiguous-MNIST — Epistemic Uncertainty")

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig("ambiguous_epistemic.png", dpi=300)

    # ============================================================
    # Aleatoric histogram
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.hist(amb_known_aleatoric, bins=100, density=True, alpha=0.6, label="Known")

    plt.hist(amb_unknown_aleatoric, bins=100, density=True, alpha=0.6, label="Unknown")

    plt.xlabel("Aleatoric uncertainty (entropy)")

    plt.ylabel("Density")

    plt.title("Ambiguous-MNIST — Aleatoric Uncertainty")

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig("ambiguous_aleatoric.png", dpi=300)


if __name__ == "__main__":
    main()
