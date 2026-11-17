from evaluation import mc_dropout_cal_sample, evaluate_student_per_class
from teacher import ConvNet_Simple
from teacher_trainer import train_teacher
from student import ConvNet_s
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from torch import nn
import ddu_dirty_mnist
import torch.nn.functional as F
import os
import pickle


def filter_digits(dataset, exclude):
    keep_indices = [i for i, (_, label) in enumerate(dataset) if label not in exclude]
    return Subset(dataset, keep_indices)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


known_unknown, unknown_unknown = 0, 9

batch_size = 32
learning_rate = 0.001
num_epochs = 60

p = [0.1, 0.05, 0.1]
LDMC = [True, "rand"]


train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)
test_ambiguous = ddu_dirty_mnist.AmbiguousMNIST(
    "./data", train=False, download=True, device="cuda"
)

train_dataset = filter_digits(train_dataset, exclude=[unknown_unknown])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
ambig_loader = DataLoader(test_ambiguous, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


for i in range(1):
    config = {
        "num_classes": 8,
        "p": p[i],
        "M": 10,
        "T": 128,
        "device": device,
        "LDMC": LDMC[i],
    }

    print(config["p"], config["LDMC"])

    if not os.path.isfile(f"{p[i]}-{LDMC[i]}-data.pkl"):
        print("Start training ****")

        teacher = ConvNet_Simple(config).to(device)
        teacher.apply(initialize_weights)

        teacher, sampler = train_teacher(
            teacher,
            config,
            known_unknown=known_unknown,
            unknown_unknown=unknown_unknown,
        )

        student = ConvNet_s(config).to(device)

        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

        criterion_unc = F.binary_cross_entropy_with_logits
        criterion_cls = nn.KLDivLoss(reduction="batchmean")

        # initialize student
        student.apply(initialize_weights)
        state_dict = teacher.model.state_dict()
        student.load_state_dict(state_dict, strict=False)

        for name, param in student.named_children():
            if name not in [
                "fc1_",
                "fc2_",
                "fc3",
                "fc4",
                "fc5",
                "fc6",
                "soft_drop1",
                "soft_drop2",
                "soft_drop3",
            ]:
                param.requires_grad = False

        gamma = 0.75

        for epoch in range(num_epochs):
            student.train()
            total_loss = 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                with torch.no_grad():
                    detail = mc_dropout_cal_sample(
                        teacher, data, sampler, T=config["T"]
                    )

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

            if (epoch + 1) % 30 == 0:
                torch.save(student.state_dict(), "./model/student_model-1.pth")
                result, _, _ = evaluate_student_per_class(
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
                )
                print(result)

            if (epoch + 1) % 30 == 0:
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

                result_amb, all_to_amb, all_al_amb = evaluate_student_per_class(
                    student,
                    teacher,
                    ambig_loader,
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
                print(result_amb)

    else:
        print("File already exists ****")
        with open(f"{p[i]}-{LDMC[i]}-data.pkl", "rb") as f:
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

        result_amb, all_to_amb, all_al_amb = evaluate_student_per_class(
            student,
            teacher,
            ambig_loader,
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
        print(result_amb)

    with open(f"{p[i]}-{LDMC[i]}.txt", "a") as f:
        print(result, file=f)
        print(result_amb, file=f)

    with open(f"{p[i]}-{LDMC[i]}-data.pkl", "wb") as f:
        obj = {
            "uncertainties": [all_to, all_al, all_to_amb, all_al_amb],
            "student": student,
            "sampler": sampler,
            "teacher": teacher,
        }
        pickle.dump(obj, f)
