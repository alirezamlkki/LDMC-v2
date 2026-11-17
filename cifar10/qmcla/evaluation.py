import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


param_map = {
    0: "features.28.weight",
    1: "features.28.bias",
    2: "fc.weight",
    3: "fc.bias",
}


def mc_cal(
    model, dataloader, allowed_indices, device, T=32, M=10, num_classes=10, sampler=None
):

    # Initialize accumulators
    class_correct = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    class_aleatoric = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    counter = False

    for data, target in tqdm(dataloader):
        data = data.to(device)

        m_indices = torch.randint(0, M, (3,))

        probs_T = sampler.predictive_samples(
            model.model, data, param_map, m_indices, n_samples=T
        )

        # Mean predictive distribution
        mean_probs = probs_T.mean(dim=0)  # [batch, num_classes]

        # --- Entropies ---
        # Aleatoric (expected entropy)
        if not counter:
            print(probs_T.shape)
            counter = True
        aleatoric_all = -(probs_T.clamp(0.00001) * probs_T.clamp(0.00001).log2()).sum(
            dim=2
        )
        aleatoric = aleatoric_all.mean(dim=0)

        # Total (predictive) entropy
        total_ent = -(mean_probs.clamp(0.00001) * mean_probs.clamp(0.00001).log2()).sum(
            dim=1
        )  # [batch]
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


def mc_dropout_cal_sample(model, batch, sampler, T=20, M=5, R=3):

    m_indices = torch.randint(0, M, (3,))

    # for _ in range(T):

    probs_T = sampler.predictive_samples(
        model.model, batch, param_map, m_indices, n_samples=T
    )
    # Mean predictive distribution
    mean_probs = probs_T.mean(dim=0)  # [batch, num_classes]

    aleatoric_all = -(probs_T.clamp(0.001) * probs_T.clamp(0.001).log2()).sum(dim=2)
    aleatoric = aleatoric_all.mean(dim=0)

    total_ent = -(mean_probs.clamp(0.00001) * mean_probs.clamp(0.00001).log2()).sum(
        dim=-1
    )  # [batch]
    epistemic = total_ent - aleatoric

    query_arr = sampler.get_masks(m_indices)

    return {
        "preds": mean_probs,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total_ent,
        "total-val": total_ent,
        "query": query_arr,
    }


@torch.no_grad()
def evaluate_student_per_class(
    student,
    teacher,
    test_loader,
    sampler,
    device,
    T=20,
    num_classes=10,
    M=5,
    R=3,
    known_unknown=2,
    unknown_unknown=3,
    roc=False,
):
    student.eval()

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

    all_to = [[] for _ in range(num_classes)]
    all_al = [[] for _ in range(num_classes)]
    all_labels = []
    all_logits = torch.tensor([], device=device)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # --- Teacher MC Dropout outputs ---
        detail = mc_dropout_cal_sample(teacher, data, sampler, T=T)
        teacher_total = detail["aleatoric"]

        m_indices = torch.randint(0, M, (3,))
        query_arr = sampler.get_masks(m_indices)

        student_output, student_unc = student(data, query_arr)
        probs = F.softmax(student_output, dim=1)
        tot_estimated = -(probs.clamp(0.0001) * probs.clamp(0.0001).log2()).sum(dim=1)
        preds = student_output.argmax(dim=1)

        prob = torch.logsumexp(student_output, dim=1)
        all_logits = torch.cat((all_logits, prob), dim=0)

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

            all_to[cls].append(tot_estimated[i].item())
            all_al[cls].append(student_unc[i].item())

            class_kl_loss[cls] += loss_func(student_pred[i], detail["preds"][i])

            if target[i].item() in [known_unknown]:
                all_labels.append(1)
            elif target[i].item() in [unknown_unknown]:
                all_labels.append(2)
            else:
                all_labels.append(0)

    # Compute averages
    acc_per_class = class_correct / np.maximum(class_total, 1)
    aleatoric_per_class = np.sqrt(class_aleatoric / np.maximum(class_total, 1))
    total_per_class = class_total_ent / np.maximum(class_total, 1)
    aleatoric_avg_per_class = class_aleatoric_avg / np.maximum(class_total, 1)
    class_kl_loss = class_kl_loss / np.maximum(class_total, 1)

    if roc:
        auroc = compute_pairwise_aurocs(all_logits, all_labels)
    else:
        auroc = None

    # Return dictionary
    return (
        {
            "accuracy_per_class": acc_per_class,
            "aleatoric_per_class": aleatoric_per_class,
            "total_per_class": total_per_class,
            "aleatoric_avg_per_class": aleatoric_avg_per_class,
            "class_kl_loss": class_kl_loss,
            "auroc": auroc,
        },
        all_to,
        all_al,
    )


@torch.no_grad()
def compute_pairwise_aurocs(scores, labels):
    """
    Compute pairwise AUROCs for:
        KK (0) vs KU (1)
        KK (0) vs UU (2)
        KU (1) vs UU (2)

    Args:
        scores: Tensor of shape (N,)
        labels: Tensor of shape (N,)
                0 = KK
                1 = KU
                2 = UU
        score_type: "energy", "msp", or "logit"

    Returns:
        dict of AUROC scores.
    """

    scores = scores.cpu().numpy()
    labels = np.array(labels)

    def pairwise_auc(pos_label, neg_label):
        mask = np.logical_or(labels == pos_label, labels == neg_label)

        binary_labels = (labels[mask] == pos_label).astype(int)
        if sum(mask) > 0:
            return roc_auc_score(binary_labels, scores[mask])
        else:
            return None

    return {
        "KK_vs_KU": pairwise_auc(0, 1),
        "KK_vs_UU": pairwise_auc(0, 2),
        "KU_vs_UU": pairwise_auc(1, 2),
    }
