import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification


class ControlledDropout(nn.Module):
    def __init__(self, p, shape, device, M=10, rw=0.5, indices=None):
        super(ControlledDropout, self).__init__()
        self.store = []
        for i in range(M):
            mask = torch.empty(shape, device=device).bernoulli_(1 - p)
            self.store.append(mask)

        self.rw = rw
        self.p = p
        self.index = indices

    def forward(self, x):

        selected = torch.stack([self.store[i] for i in self.index[0]], dim=0)
        random_walk = torch.ones_like(selected, device=x.device) * self.rw
        random_walk = random_walk + selected * (2 * self.p - 1) / (1 - self.p) * self.rw
        random_walk = torch.bernoulli(random_walk)

        mask = (selected.int() ^ random_walk.int()) / (1 - self.p)

        return x * mask

    def get_candidates(self, indices):
        return [self.store[i] for i in indices]


class TeacherModel(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.base = DistilBertForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=config["num_classes"]
        )

        self.mask_indices = [None]
        shape = [64, 768]

        self.base.distilbert.embeddings.dropout = ControlledDropout(
            config["p"],
            shape,
            config["device"],
            M=config["M"],
            rw=config["rw"],
            indices=self.mask_indices,
        )

        for i, layer in enumerate(self.base.distilbert.transformer.layer):
            layer.attention.dropout = ControlledDropout(
                config["p"],
                shape,
                config["device"],
                M=config["M"],
                rw=config["rw"],
                indices=self.mask_indices,
            )
            layer.ffn.dropout = ControlledDropout(
                config["p"],
                shape,
                config["device"],
                M=config["M"],
                rw=config["rw"],
                indices=self.mask_indices,
            )

        self.base.dropout = ControlledDropout(
            config["p"],
            [768],
            config["device"],
            M=config["M"],
            rw=config["rw"],
            indices=self.mask_indices,
        )

    def forward(self, x, att, mask_indices):

        self.mask_indices.pop()
        self.mask_indices.append(mask_indices)

        output = self.base(x, attention_mask=att)

        return output.logits

    def get_candidates(self, indices):
        masks = {}

        masks["dropout_emb"] = torch.stack(
            self.base.distilbert.embeddings.dropout.get_candidates(indices)
        )

        masks["dropout_att"] = {}
        masks["dropout_ffn"] = {}
        for i, layer in enumerate(self.base.distilbert.transformer.layer):
            masks["dropout_att"][i] = torch.stack(
                layer.attention.dropout.get_candidates(indices)
            )
            masks["dropout_ffn"][i] = torch.stack(
                layer.ffn.dropout.get_candidates(indices)
            )

        masks["dropout_cls"] = torch.stack(self.base.dropout.get_candidates(indices))
        masks["dropout_cls"] = masks["dropout_cls"].unsqueeze(dim=1)

        return masks
