import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from evaluation import mc_dropout_cal_sample, evaluate_student_per_class
from teacher_trainer import train_teacher
import torch.nn as nn
from student import vgg16_cifar_student
from student import SoftDropout
from teacher import vgg16_cifar_teacher


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


config = {
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
        "P",
        256,
        256,
        256,
        "M",
        "P",
    ],
    "num_classes": 8,
    "p": 0.3,
    "rw": 0.2,
    "M": 10,
    "T": 128,
    "R": 3,
    "device": device,
    "LDMC": True,
}


batch_size = 32
learning_rate = 0.001
num_epochs = 40

known_unknown, unknown_unknown = 2, 3


teacher = vgg16_cifar_teacher(config).to(device)

teacher.apply(initialize_weights)

train_teacher(
    teacher, config, known_unknown=known_unknown, unknown_unknown=unknown_unknown
)
student = vgg16_cifar_student(config).to(device)


transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)


def filter_digits(dataset, exclude):
    keep_indices = [i for i, (_, label) in enumerate(dataset) if label not in exclude]
    return Subset(dataset, keep_indices)


train_dataset = filter_digits(train_dataset, exclude=[unknown_unknown])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


optimizer = torch.optim.SGD(
    student.parameters(),
    lr=0.1,  # initial learning rate
    momentum=0.9,  # helps smooth gradients
    weight_decay=5e-4,  # important for regularization
)


scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 40, 60], gamma=0.1
)


criterion_unc = F.binary_cross_entropy_with_logits
criterion_cls = nn.KLDivLoss(reduction="batchmean")


# initialize the student
student.apply(initialize_weights)
state_dict = teacher.features.state_dict()

student.features.load_state_dict(state_dict, strict=False)
for m in student.features.modules():
    if not isinstance(m, SoftDropout):
        for p in m.parameters():
            p.requires_grad = False


gamma = 0.75

for epoch in range(num_epochs):
    student.train()
    total_loss = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            detail = mc_dropout_cal_sample(teacher, data, T=config["T"])

        optimizer.zero_grad()
        output, unc = student(data, detail["masks"])
        output = F.log_softmax(output, dim=-1)
        loss_cls = criterion_cls(output, detail["preds"])
        loss_unc = F.mse_loss(unc.squeeze(), detail["aleatoric"])

        loss = loss_cls + gamma * loss_unc

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    if (epoch + 1) % 20 == 0:
        torch.save(student.state_dict(), "./model/student_model-1.pth")
        print(
            evaluate_student_per_class(
                student,
                teacher,
                train_loader,
                device,
                T=config["T"],
                num_classes=10,
                known_unknown=known_unknown,
                unknown_unknown=unknown_unknown,
                M=config["M"],
                R=config["R"],
            )
        )

    if (epoch + 1) % 20 == 0:
        print(
            evaluate_student_per_class(
                student,
                teacher,
                test_loader,
                device,
                T=config["T"],
                num_classes=10,
                known_unknown=known_unknown,
                unknown_unknown=unknown_unknown,
                M=config["M"],
                R=config["R"],
            )
        )

print("*****************---------*****************")
