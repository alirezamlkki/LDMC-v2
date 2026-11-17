import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, random_split, Dataset
from evaluation import mc_cal
import os
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

    T = config["T"]
    M = config["M"]
    p_m = config["p"]

    sv_path = "./model/teacher_model.pth"

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    allowed_indiced = []
    for i in range(10):
        if i not in [known_unknown, unknown_unknown]:
            allowed_indiced.append(i)

    batch_size = 64
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    def filter_digits(dataset, exclude):
        keep_indices = [
            i for i, (_, label) in enumerate(dataset) if label not in exclude
        ]
        return Subset(dataset, keep_indices)

    train_dataset = filter_digits(
        train_dataset, exclude=[known_unknown, unknown_unknown]
    )

    val_size = int(len(train_dataset) * 0.10)
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    train_dataset = TrainDataset(train_dataset, allowed_indiced)
    val_dataset = TrainDataset(val_dataset, allowed_indiced)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if not os.path.exists(sv_path):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,  # initial learning rate
            momentum=0.9,  # helps smooth gradients
            weight_decay=5e-4,  # important for regularization
        )

        schedule = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 70], gamma=0.1
        )
        criterion = nn.CrossEntropyLoss()

        print("*** Start teacher training")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
            )
            schedule.step()

        torch.save(model.state_dict(), sv_path)

    else:
        state_dict = torch.load(sv_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

    model.eval()
    model = model.fuse_bn()
    model.train()

    for module in model.features.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            for p in module.parameters():
                p.requires_grad = False
        else:
            for p in module.parameters(recurse=False):
                p.requires_grad = False

    conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
    for layer in conv_layers[-1:]:
        for p in layer.parameters():
            p.requires_grad = True

    # classifier (nn.Linear) has no BN after it -- always safe, keep it unfrozen
    # for p in model.classifier.parameters():

    for p in model.fc.parameters():
        p.requires_grad = True

    model = Laplace(
        model, "classification", subset_of_weights="all", hessian_structure="kron"
    )

    model.fit(train_loader)

    # for attr in dir(model):
    #     if not attr.startswith("_"):

    model.optimize_prior_precision(
        method="gridsearch",
        pred_type="nn",
        link_approx="mc",
        val_loader=val_loader,
        n_samples=T,
    )

    sampler = KronSpectralSampler(model, M=M, p=p_m, seed=42)
    prec = 25.0

    model.prior_precision = torch.tensor(prec, device=device)
    with torch.no_grad():
        result = mc_cal(
            model, test_loader, allowed_indiced, device, T=T, M=M, sampler=sampler
        )
        print(result)

    return model, sampler
