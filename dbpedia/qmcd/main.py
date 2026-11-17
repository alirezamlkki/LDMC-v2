import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from evaluation import mc_dropout_cal_sample, evaluate_student_per_class
from teacher_trainer import train_teacher
import torch.nn as nn
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from tqdm import tqdm
from student import StudentModel
from teacher import TeacherModel


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].unsqueeze(-2).bool(),
            "label": torch.tensor(label, dtype=torch.long),
        }


known_unknown = 12
unknown_unknown = 13
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "model_name": "distilbert-base-uncased",
    "num_classes": 12,
    "p": 0.5,
    "rw": 0.4,
    "M": 10,
    "T": 64,
    "R": 3,
    "device": device,
    "LCMC": True,
}

num_epochs = 2


LCMC = [True, False]


for j in range(2):
    print(config)
    for _ in range(1):
        config["LCMC"] = LCMC[j]
        allowed_indiced = []
        for i in range(14):
            if i not in [unknown_unknown]:
                allowed_indiced.append(i)

        teacher = TeacherModel(config).to(device)

        for name, param in teacher.named_parameters():
            if name in teacher.state_dict().keys():
                if (
                    ("classifier" not in name)
                    and ("dropout" not in name)
                    and (name not in ["fc3", "fc4", "fc5", "fc6", "fc7"])
                ):
                    param.requires_grad = False

        print("******** train teacher")
        train_teacher(
            teacher,
            config,
            known_unknown=known_unknown,
            unknown_unknown=unknown_unknown,
        )

        continue
        student = StudentModel(config).to(device)

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        dataset = load_dataset("dbpedia_14")

        train_full = dataset["train"]
        train_full = TextDataset(
            train_full["content"], train_full["label"], tokenizer, max_len=64
        )
        # Keep only labels 1..8 for training
        train_indices = [
            i for i, data in enumerate(train_full) if data["label"] in allowed_indiced
        ]
        train_full = Subset(train_full, train_indices)

        test_ds = dataset["test"]
        test_ds = TextDataset(
            test_ds["content"], test_ds["label"], tokenizer, max_len=64
        )

        student.load_state_dict(teacher.state_dict(), strict=False)

        for name, param in student.named_parameters():
            if name in teacher.state_dict().keys():
                if (
                    ("classifier" not in name)
                    and ("dropout" not in name)
                    and (name not in ["fc3", "fc4", "fc5", "fc6", "fc7"])
                ):
                    param.requires_grad = False

        train_loader = DataLoader(
            train_full,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=2e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        criterion_cls = nn.KLDivLoss(reduction="batchmean")

        gamma = 0.75

        print("******** train student")

        for epoch in range(num_epochs):
            student.train()
            total_loss = 0

            for data in tqdm(train_loader):
                x = data["input_ids"].to(device)
                y = data["label"].to(device)
                att = data["attention_mask"].to(device)

                with torch.no_grad():
                    detail = mc_dropout_cal_sample(teacher, data, T=config["T"])

                optimizer.zero_grad()
                output, unc = student(x, att, detail["masks"])

                output = F.log_softmax(output, dim=-1)
                loss_cls = criterion_cls(output, detail["preds"])
                loss_unc = F.mse_loss(unc.squeeze(), detail["aleatoric"])

                loss = loss_cls + gamma * loss_unc

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            with open(
                f"log-{known_unknown}{unknown_unknown}-{config['p']}-{config['rw']}.txt",
                "a",
            ) as f:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}",
                    file=f,
                )
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
            )

            if (epoch + 1) % 1 == 0:
                torch.save(student.state_dict(), "./model/student_model-1.pth")
                res = evaluate_student_per_class(
                    student,
                    teacher,
                    train_loader,
                    device,
                    T=config["T"],
                    num_classes=14,
                    known_unknown=known_unknown,
                    unknown_unknown=unknown_unknown,
                    M=config["M"],
                    R=config["R"],
                )
                with open(
                    f"log-{known_unknown}{unknown_unknown}-{config['p']}-{config['rw']}.txt",
                    "a",
                ) as f:
                    print(res, file=f)
                print(res)

            if (epoch + 1) % 1 == 0:
                res = evaluate_student_per_class(
                    student,
                    teacher,
                    test_loader,
                    device,
                    T=config["T"],
                    num_classes=14,
                    known_unknown=known_unknown,
                    unknown_unknown=unknown_unknown,
                    M=config["M"],
                    R=config["R"],
                )
                with open(
                    f"log-{known_unknown}{unknown_unknown}-{config['p']}-{config['rw']}.txt",
                    "a",
                ) as f:
                    print(res, file=f)
                print(res)

        print("*****************---------*****************")
