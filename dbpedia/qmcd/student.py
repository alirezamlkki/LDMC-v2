import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification


class ControlledDropout(nn.Module):
    def __init__(self, p, ref, LCMC=True):
        super(ControlledDropout, self).__init__()

        self.agg = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        self.masks = ref
        self.p = p
        self.LCMC = LCMC

    def forward(self, x):

        if not self.LCMC:
            return x

        mask = self.agg(self.masks[0])
        mask = F.sigmoid(mask) / (1 - self.p)

        if mask.shape[0] == 1 and mask.shape[1] == 1:
            mask = mask.squeeze(dim=0)

        return x * mask


class StudentModel(nn.Module):
    def __init__(self, config, unc_num=7, T=32):

        super().__init__()

        self.base = DistilBertForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=config["num_classes"]
        )

        self.mask_storage = {
            "dropout_emb": [],
            "dropout_att": {
                i: [] for i in range(len(self.base.distilbert.transformer.layer))
            },
            "dropout_ffn": {
                i: [] for i in range(len(self.base.distilbert.transformer.layer))
            },
            "dropout_cls": [],
        }

        p = config["p"]

        self.base.distilbert.embeddings.dropout = ControlledDropout(
            p, ref=self.mask_storage["dropout_emb"], LCMC=config["LCMC"]
        )

        for i, layer in enumerate(self.base.distilbert.transformer.layer):
            layer.attention.dropout = ControlledDropout(
                p, ref=self.mask_storage["dropout_att"][i], LCMC=config["LCMC"]
            )
            layer.ffn.dropout = ControlledDropout(
                p, ref=self.mask_storage["dropout_ffn"][i], LCMC=config["LCMC"]
            )

        self.base.dropout = ControlledDropout(
            p, ref=self.mask_storage["dropout_cls"], LCMC=config["LCMC"]
        )

        self.fc3 = nn.Linear(768, 128)
        self.fc4 = nn.Linear(128, 8)
        self.fc5 = nn.Linear(20, 20)
        self.fc6 = nn.Linear(20, 10)

        self.fc7 = nn.Linear(10, 1)
        self.LCMC = config["LCMC"]

    def forward(self, x, att, mask=None):

        for key, val in self.mask_storage.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    if len(v):
                        v.pop()
            else:
                if len(val):
                    self.mask_storage[key].pop()

        if mask is not None:
            for key, val in mask.items():
                if isinstance(val, dict):
                    for k, ref in mask[key].items():
                        if len(ref):
                            self.mask_storage[key][k].append(ref)
                else:
                    if len(val):
                        self.mask_storage[key].append(val)

        outputs = self.base.distilbert(input_ids=x, attention_mask=att)

        hidden_states = outputs.last_hidden_state  # (B, S, H)

        # CLS token
        pooled = hidden_states[:, 0]  # (B, H)

        # Original classifier path
        pooled_cls = self.base.pre_classifier(pooled)
        pooled_cls = F.relu(pooled_cls)

        pooled_cls = self.base.dropout(pooled_cls)
        logits = self.base.classifier(pooled_cls)

        # ---- Your uncertainty head ----
        unc_hidden = F.relu(self.fc3(pooled))
        unc_hidden = self.fc4(unc_hidden)

        combined = torch.cat((unc_hidden, logits), dim=-1)
        combined = F.silu(combined)

        combined = F.relu(self.fc5(combined))
        combined = F.relu(self.fc6(combined))

        unc = F.softplus(self.fc7(combined))

        return logits, unc
