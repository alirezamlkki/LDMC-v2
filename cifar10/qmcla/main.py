import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from evaluation import mc_dropout_cal_sample, evaluate_student_per_class
from teacher_trainer import train_teacher
import torch.nn as nn
from student import vgg16_cifar_student
from student import Attention
from teacher import vgg16_cifar_teacher
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def filter_digits(dataset, exclude):
    keep_indices = [i for i, (_, label) in enumerate(dataset) if label not in exclude]
    return Subset(dataset, keep_indices)


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


batch_size = 32
learning_rate = 0.001
num_epochs = 40
gamma = 0.75
known_unknown, unknown_unknown = 2, 3

train_dataset = filter_digits(train_dataset, exclude=[unknown_unknown])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

sv_path = "model/teacher_model.pth"

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
        256,
        256,
        128,
        "P",
    ],
    "num_classes": 8,
    "p": 0.8,
    "M": 5,
    "T": 128,
    "device": device,
    "LDMC": False,
}

p_s = [0.05, 0.05]
LDMC = [False, True]


for i in range(2):
    config["p"] = p_s[i]
    config["LDMC"] = LDMC[i]

    print(config["p"], config["LDMC"])
    if not os.path.isfile(f"{p_s[i]}-{LDMC[i]}-data.pkl"):
        print("Start training ****")

        teacher = vgg16_cifar_teacher(config).to(device)

        teacher.apply(initialize_weights)

        teacher, sampler = train_teacher(
            teacher,
            config,
            known_unknown=known_unknown,
            unknown_unknown=unknown_unknown,
        )

        student = vgg16_cifar_student(config).to(device)

        optimizer = torch.optim.SGD(
            student.parameters(),
            lr=0.1,  # initial learning rate
            momentum=0.9,  # helps smooth gradients
            weight_decay=5e-4,  # important for regularization
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20], gamma=0.1
        )

        criterion_unc = F.binary_cross_entropy_with_logits
        criterion_cls = nn.KLDivLoss(reduction="batchmean")

        # initialize the student
        student.apply(initialize_weights)
        state_dict = torch.load(sv_path, map_location=device)

        teacher_tmp = vgg16_cifar_teacher(config).to(device)
        teacher_tmp.load_state_dict(state_dict)
        student.features.load_state_dict(
            teacher_tmp.features.state_dict(), strict=False
        )
        del teacher_tmp

        for j, m in enumerate(student.features.modules()):
            if not isinstance(m, Attention) and j < 20:
                for p in m.parameters():
                    p.requires_grad = False

        for epoch in range(num_epochs):
            student.train()
            total_loss = 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                with torch.no_grad():
                    detail = mc_dropout_cal_sample(
                        teacher, data, sampler, T=config["T"], M=config["M"]
                    )

                optimizer.zero_grad()
                output, unc = student(data, detail["query"])

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

            if (epoch + 1) % 20 == 0:
                torch.save(student.state_dict(), "./model/student_model-1.pth")
                resutl, _, _ = evaluate_student_per_class(
                    student,
                    teacher,
                    train_loader,
                    sampler,
                    device,
                    T=config["T"],
                    num_classes=10,
                    known_unknown=known_unknown,
                    unknown_unknown=unknown_unknown,
                    M=config["M"],
                    roc=False,
                )
                print(resutl)

            if (epoch + 1) % 20 == 0:
                result, all_to, all_al = evaluate_student_per_class(
                    student,
                    teacher,
                    test_loader,
                    sampler,
                    device,
                    T=config["T"],
                    num_classes=10,
                    known_unknown=known_unknown,
                    unknown_unknown=unknown_unknown,
                    M=config["M"],
                    roc=True,
                )
                with open(f"{config['p']}-{config['LDMC']}.txt", "a") as f:
                    print(result)
                    print(result, file=f)
                    print(f"{epoch + 1}----------------***********", file=f)

        print("*****************---------*****************")

    else:
        print("File already exists ****")
        with open(f"{p_s[i]}-{LDMC[i]}-data.pkl", "rb") as f:
            obj = pickle.load(f)
            student = obj["student"]
            teacher = obj["teacher"]
            sampler = obj["sampler"]

        result, all_to, all_al = evaluate_student_per_class(
            student,
            teacher,
            test_loader,
            sampler,
            device,
            T=config["T"],
            num_classes=10,
            known_unknown=known_unknown,
            unknown_unknown=unknown_unknown,
            M=config["M"],
            roc=True,
        )

        print(result)

    with open(f"{p_s[i]}-{LDMC[i]}.txt", "a") as f:
        print(result, file=f)

    with open(f"{p_s[i]}-{LDMC[i]}-data.pkl", "wb") as f:
        obj = {
            "uncertainties": [all_to, all_al, None, None],
            "student": student,
            "sampler": sampler,
            "teacher": teacher,
        }
        pickle.dump(obj, f)
