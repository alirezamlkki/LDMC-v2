import torch
import torch.nn as nn
import ddu_dirty_mnist
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, random_split, Dataset
from evaluation import mc_cal_auroc
from laplace import Laplace
from teacher_laplace import KronSpectralSampler


class TrainDataset(Dataset):
    def __init__(self, base_dataset, allowed_indiced):
        self.base_dataset = base_dataset
        self.allowed_indiced = allowed_indiced

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        target = self.allowed_indiced.index(target)
        return data, target


def train_teacher(model, config, known_unknown=0, unknown_unknown=1):

    M = config["M"]
    p_m = config["p"]
    T = config["T"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # mean, std of MNIST
        ]
    )

    allowed_indiced = []
    for i in range(10):
        if i not in [known_unknown, unknown_unknown]:
            allowed_indiced.append(i)

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    device = config["device"]

    full_train = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    test_ambiguous = ddu_dirty_mnist.AmbiguousMNIST(
        "./data", train=False, download=True, device="cuda"
    )

    train_dataset, val_dataset = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )

    def filter_digits(dataset, exclude):
        keep_indices = [
            i for i, (_, label) in enumerate(dataset) if label not in exclude
        ]
        return Subset(dataset, keep_indices)

    train_dataset = filter_digits(
        train_dataset, exclude=[known_unknown, unknown_unknown]
    )
    train_dataset = TrainDataset(train_dataset, allowed_indiced)
    val_dataset = filter_digits(val_dataset, exclude=[known_unknown, unknown_unknown])
    val_dataset = TrainDataset(val_dataset, allowed_indiced)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    ambig_loader = DataLoader(test_ambiguous, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in train_loader:
            index = torch.randint(0, M, [len(target)])

            data, target = data.to(device), target.to(device)

            # for i in range(len(target)):

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
            )
        schedule.step()

    for p in model.fc2.parameters():
        p.requires_grad = False

    for p in model.conv1.parameters():
        p.requires_grad = False

    for p in model.conv2.parameters():
        p.requires_grad = False

    model = Laplace(
        model, "classification", subset_of_weights="all", hessian_structure="kron"
    )

    model.fit(train_loader)

    model.optimize_prior_precision(
        method="gridsearch",
        pred_type="nn",
        link_approx="mc",
        val_loader=valid_loader,
        n_samples=T,
    )

    sampler = KronSpectralSampler(model, M=M, p=p_m, seed=42)
    prec = 30.0

    model.prior_precision = torch.tensor(prec, device=device)
    with torch.no_grad():
        result = mc_cal_auroc(
            model, test_loader, allowed_indiced, T=T, M=M, sampler=sampler
        )
        result_amb = mc_cal_auroc(
            model, ambig_loader, allowed_indiced, T=T, M=M, sampler=sampler
        )
        print(result)
        print(result_amb)

    return model, sampler
