import torch.nn.functional as F
import numpy as np
from torch import nn
import torch
from tqdm import tqdm


def mc_cal(model, dataloader, allowed_indices, T=32, M=10, R=3, num_classes=10):
    model.train()  # Keep dropout active
    device = next(model.parameters()).device

    # Initialize accumulators
    class_correct = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    class_aleatoric = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    loop = tqdm(dataloader, desc="test", leave=False)
    for data in loop:
        x = data["input_ids"].to(device)
        target = data["label"].to(device)
        att = data["attention_mask"].to(device)
        batch_size = x.size(0)

        # Monte Carlo samples
        candidates = torch.randperm(M)[:R].to(device)
        probs_T = []

        for _ in range(T):
            index = candidates[torch.randint(0, R, [len(target)])]
            out = model(x, att, index)
            probs = F.softmax(out, dim=1)
            probs_T.append(probs.unsqueeze(0))  # [1, batch, num_classes]
        probs_T = torch.cat(probs_T, dim=0)  # [T, batch, num_classes]

        probs_T_s = probs_T / probs_T.sum(dim=-1, keepdim=True)
        mean_probs_s = probs_T_s.mean(dim=0)

        # Mean predictive distribution
        mean_probs = probs_T.mean(dim=0)  # [batch, num_classes]

        # --- Entropies ---
        # Aleatoric (expected entropy)

        aleatoric_all = -(
            probs_T_s.clamp(0.00001) * probs_T_s.clamp(0.00001).log2()
        ).sum(dim=2)
        aleatoric = aleatoric_all.mean(dim=0)

        # Total (predictive) entropy
        total_ent = -(
            mean_probs_s.clamp(0.00001) * mean_probs_s.clamp(0.00001).log2()
        ).sum(dim=1)  # [batch]
        # Epistemic = total - aleatoric
        epistemic = total_ent - aleatoric

        preds = mean_probs.argmax(dim=1)

        for i in range(len(target)):
            cls = target[i].item()

            class_counts[cls] += 1
            class_aleatoric[cls] += aleatoric[i].item()
            class_total[cls] += total_ent[i].item()
            class_correct[cls] += (allowed_indices[preds[i]] == target[i]).item()

    # Compute averages
    avg_acc = class_correct / np.maximum(class_counts, 1)
    avg_aleatoric = class_aleatoric / np.maximum(class_counts, 1)
    avg_total = class_total / np.maximum(class_counts, 1)
    avg_epistemic = avg_total - avg_aleatoric

    return {
        "accuracy_per_class": avg_acc,
        "aleatoric_per_class": avg_aleatoric,
        "total_per_class": avg_total,
        "epistemic_per_class": avg_epistemic,
    }


def mc_dropout_cal_sample(model, batch, T=20, M=10, R=3):
    model.train()  # keep dropout active

    device = next(model.parameters()).device

    probs_T = []

    x = batch["input_ids"].to(device)
    att = batch["attention_mask"].to(device)

    batch_size = x.size(0)
    candidates = torch.randperm(M)[:R].to(device)
    for _ in range(T):
        index = candidates[torch.randint(0, R, [batch_size])]
        out = model(x, att, index)
        probs = F.softmax(out, dim=1)
        probs_T.append(probs.unsqueeze(0))  # [1, batch, num_classes]
    probs_T = torch.cat(probs_T, dim=0)  # [T, batch, num_classes]

    sort_cands, _ = torch.sort(candidates)
    masks = model.get_candidates(sort_cands)

    # Mean predictive distribution
    mean_probs = probs_T.mean(dim=0)  # [batch, num_classes]

    aleatoric_all = -(probs_T.clamp(0.001) * probs_T.clamp(0.001).log2()).sum(dim=-1)

    aleatoric = aleatoric_all.mean(dim=0)

    total_ent = -(mean_probs.clamp(0.00001) * mean_probs.clamp(0.00001).log2()).sum(
        dim=-1
    )  # [batch]
    epistemic = total_ent - aleatoric

    return {
        "preds": mean_probs,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total_ent,
        "total-val": total_ent,
        "masks": masks,
    }


@torch.no_grad()
def evaluate_student_per_class(
    student,
    teacher,
    test_loader,
    device,
    T=20,
    num_classes=10,
    M=10,
    R=3,
    known_unknown=0,
    unknown_unknown=1,
):
    student.eval()
    teacher.train()

    allowed_indices = []
    for i in range(num_classes):
        if i not in [known_unknown, unknown_unknown]:
            allowed_indices.append(i)

    # Initialize accumulators
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    class_aleatoric = np.zeros(num_classes)
    class_aleatoric_avg = np.zeros(num_classes)
    class_total_ent = np.zeros(num_classes)
    class_kl_loss = np.zeros(num_classes)

    loss_func = nn.KLDivLoss(reduction="batchmean")

    all_unc = []
    all_labels = []

    for data in test_loader:
        x = data["input_ids"].to(device)
        target = data["label"].to(device)
        att = data["attention_mask"].to(device)

        # --- Teacher MC Dropout outputs ---
        detail = mc_dropout_cal_sample(teacher, data, T=T)
        teacher_total = detail["aleatoric"]

        cnadidates, _ = torch.sort(torch.randperm(M)[:R].to(device))
        masks = teacher.get_candidates(cnadidates)

        student_output, student_unc = student(x, att, masks)
        preds = student_output.argmax(dim=1)

        student_pred = F.log_softmax(student_output, dim=1)

        for i in range(len(target)):
            cls = target[i].item()

            preds[i] = allowed_indices[preds[i]]

            class_total[cls] += 1
            class_correct[cls] += (preds[i] == target[i]).item()
            class_aleatoric[cls] += np.power(
                teacher_total[i].item() - student_unc[i].item(), 2
            )
            class_aleatoric_avg[cls] += student_unc[i].item()
            class_total_ent[cls] += teacher_total[i].item()
            class_kl_loss[cls] += loss_func(student_pred[i], detail["preds"][i])

            if target[i].item() == unknown_unknown:
                all_unc.append(student_unc[i].item())
                all_labels.append(teacher_total[i].item())

    # Compute averages
    acc_per_class = class_correct / np.maximum(class_total, 1)
    aleatoric_per_class = np.sqrt(class_aleatoric / np.maximum(class_total, 1))
    total_per_class = class_total_ent / np.maximum(class_total, 1)
    aleatoric_avg_per_class = class_aleatoric_avg / np.maximum(class_total, 1)
    class_kl_loss = class_kl_loss / np.maximum(class_total, 1)

    # Return dictionary
    return {
        "accuracy_per_class": acc_per_class,
        "aleatoric_per_class": aleatoric_per_class,
        "total_per_class": total_per_class,
        "aleatoric_avg_per_class": aleatoric_avg_per_class,
        "class_kl_loss": class_kl_loss,
    }
