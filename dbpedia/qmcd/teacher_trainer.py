import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from transformers import DistilBertTokenizerFast
from datasets import load_dataset
from evaluation import mc_cal


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
            "attention_mask": encoding["attention_mask"].squeeze(0).bool(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train_teacher(model, config, known_unknown=0, unknown_unknown=1):

    allowed_indiced = []
    for i in range(14):
        if i not in [known_unknown, unknown_unknown]:
            allowed_indiced.append(i)

    print(allowed_indiced)
    batch_size = 64
    num_epochs = 1
    T = config["T"]
    M = config["M"]
    R = config["R"]
    p = config["p"]
    rw = config["rw"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    dataset = load_dataset("dbpedia_14")

    train_full = dataset["train"]
    train_full = TextDataset(
        train_full["content"], train_full["label"], tokenizer, max_len=64
    )

    test_ds = dataset["test"]
    test_ds = TextDataset(test_ds["content"], test_ds["label"], tokenizer, max_len=64)

    train_indices = [
        i for i, data in enumerate(train_full) if data["label"] in allowed_indiced
    ]
    train_ds = Subset(train_full, train_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=2, pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    criterion = nn.CrossEntropyLoss()

    accuracy = Accuracy(task="multiclass", num_classes=12).to(device)

    for epoch in range(1, num_epochs + 1):
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        running_loss, running_acc = 0.0, 0.0
        for data in loop:
            x = data["input_ids"].to(device)
            y = data["label"].to(device)
            att = data["attention_mask"].to(device)
            index = torch.randint(0, M, [len(y)])

            for i in range(y.shape[0]):
                y[i] = allowed_indiced.index(y[i])

            optimizer.zero_grad()
            logits = model(x, att, index)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_acc += accuracy(logits, y) * x.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)

        print(
            f"Epoch {epoch:02d}/{num_epochs} - train loss: {epoch_loss:.4f} - train acc: {epoch_acc:.2f}"
        )
        scheduler.step()
        torch.save(model.state_dict(), "./model/teacher_model.pth")

    model.train()
    with torch.no_grad():
        result = mc_cal(
            model, test_loader, allowed_indiced, T=T, M=M, R=R, num_classes=14
        )
        with open(f"log-{known_unknown}{unknown_unknown}-{p}-{rw}.txt", "a") as f:
            print(result, file=f)
        print(result)

    torch.save(model.state_dict(), "./model/teacher_model.pth")
