from evaluation import mc_dropout_cal_sample, evaluate_student_per_class
from teacher import ConvNet
from teacher_trainer import train_teacher
from student import ConvNet_s
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "num_classes": 8,
    "p": 0.5,
    "rw": 0.2,
    "M": 10,
    "T": 128,
    "R": 3,
    "device": device,
    "LDMC": False,
}

batch_size = 32
learning_rate = 0.001
num_epochs = 40


known_unknown, unknown_unknown = 0, 9


for _ in range(2):
    teacher = ConvNet(config).to(device)
    teacher.apply(initialize_weights)

    train_teacher(
        teacher, config, known_unknown=known_unknown, unknown_unknown=unknown_unknown
    )

    student = ConvNet_s(config).to(device)

    def filter_digits(dataset, exclude):
        keep_indices = [
            i for i, (_, label) in enumerate(dataset) if label not in exclude
        ]
        return Subset(dataset, keep_indices)

    train_dataset = filter_digits(train_dataset, exclude=[unknown_unknown])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    criterion_unc = F.binary_cross_entropy_with_logits
    criterion_cls = nn.KLDivLoss(reduction="batchmean")

    # initialize student
    student.apply(initialize_weights)
    state_dict = teacher.state_dict()
    student.load_state_dict(state_dict, strict=False)

    # for name, param in student.named_children():
    #     if name not in ["fc1_", "fc2_", "fc3", "fc4", "fc5", "fc6",  "soft_drop1", "soft_drop2"]:

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
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
        )

        if (epoch + 1) % 10 == 0:
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

        if (epoch + 1) % 10 == 0:
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
    print("***********----------------")
