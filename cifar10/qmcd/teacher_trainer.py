import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from evaluation import mc_cal


def train_teacher(model, config, known_unknown=0, unknown_unknown=1):

    T = config["T"]
    M = config["M"]
    R = config["R"]

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
    num_epochs = 100
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
        )
        schedule.step()

    model.train()
    with torch.no_grad():
        result = mc_cal(model, test_loader, allowed_indiced, T=T, M=M, R=R)
        print(result)

    torch.save(model.state_dict(), "./model/teacher_model.pth")
