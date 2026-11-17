import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from evaluation import mc_cal_auroc
import ddu_dirty_mnist


def train_teacher(model, config, known_unknown=0, unknown_unknown=1):

    M = config["M"]
    R = config["R"]
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
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_ambiguous = ddu_dirty_mnist.AmbiguousMNIST(
        "./data", train=False, download=True, device="cuda"
    )

    def filter_digits(dataset, exclude):
        keep_indices = [
            i for i, (_, label) in enumerate(dataset) if label not in exclude
        ]
        return Subset(dataset, keep_indices)

    train_dataset = filter_digits(
        train_dataset, exclude=[known_unknown, unknown_unknown]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    ambig_loader = DataLoader(test_ambiguous, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            index = torch.randint(0, M, [len(target)])

            data, target = data.to(device), target.to(device)

            for i in range(len(target)):
                target[i] = allowed_indiced.index(target[i])

            optimizer.zero_grad()
            output = model(data, index)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
            )
        schedule.step()

    model.train()
    with torch.no_grad():
        result = mc_cal_auroc(model, test_loader, allowed_indiced, T=T, M=M, R=R)
        print(result)

        result = mc_cal_auroc(model, ambig_loader, allowed_indiced, T=T, M=M, R=R)
        print(result)

    torch.save(model.state_dict(), "./model/teacher_model.pth")
