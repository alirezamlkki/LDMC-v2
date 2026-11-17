import torch
import numpy as np


class KronSpectralSampler:
    """
    Reads the fitted K-FAC eigendecomposition directly from
    la.posterior_precision (used ONLY for eigenvectors/masks, not for
    reconstructing eigenvalue scale by hand -- that reimplementation is what
    caused the delta_W blow-up/divergence).

    For actual weight perturbations, draws a real sample from la.sample(),
    which uses the library's own correctly-regularized posterior. The
    spectral mask is then applied to that already-correctly-scaled delta,
    not to a hand-rebuilt covariance.
    """

    def __init__(self, la_model, M=10, p=0.1, seed=42):
        self.la_model = la_model
        self.M = M
        self.p = float(p)
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.decomposed = {}
        self._decompose_once()
        la_model._spectral_cache = self

    # ---------- one-time work: cache eigenvectors + masks only ----------
    def _decompose_once(self):
        """
        We only need eigenvectors (fixed rotation basis) and masks here.
        Eigenvalue-derived scale is intentionally NOT reconstructed by hand
        -- la.sample() already applies the library's correct l1⊗l2 + delta
        regularization internally.
        """
        post_prec = self.la_model.posterior_precision

        for i, evecs_i in enumerate(post_prec.eigenvectors):
            self.decomposed[i] = {}
            for j, eigvecs in enumerate(evecs_i):
                k = eigvecs.shape[1] if eigvecs.dim() > 1 else eigvecs.shape[0]
                masks = torch.bernoulli(
                    torch.full((self.M, k), 1 - self.p, device=eigvecs.device)
                )
                self.decomposed[i][j] = {
                    "eigenvectors": eigvecs,
                    "masks": masks,
                }

    def n_factors(self, i):
        return len(self.decomposed[i])

    def get_masks(self, mc_idices):
        """
        Return the masks applied at this MC iteration, keyed by
        layer index i and kron-factor index j: masks[i][j] -> (k,) tensor.
        """
        all_masks = []
        for ind in mc_idices:
            masks = {}
            for i, layer in self.decomposed.items():
                masks[i] = {}
                for j, entry in layer.items():
                    masks[i][j] = entry["masks"][ind]
            all_masks.append(masks)

        return all_masks

    # ---------- draw a real, correctly-scaled sample from Laplace ----------
    def _build_offsets(self, base_model, param_map):
        """
        Maps each param_map name to its (start, end) offset in the flat
        parameter vector used by la.mean / la.sample(), based on the
        model's own named_parameters() order (must match what Laplace
        used internally -- i.e. the order of parameters with requires_grad=True).
        """
        offsets = {}
        offset = 0
        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            if name in param_map.values():
                offsets[name] = (offset, offset + numel)
            offset += numel
        return offsets

    def masked_forward_via_la_sample(
        self, base_model, x, param_map, mc_idx, offsets=None
    ):
        """
        Draw one real sample from la.sample() (correctly regularized),
        extract delta_W = sample - MAP per targeted parameter, project that
        delta through the eigenbasis, zero out entries per the spectral
        mask for this mc_idx, project back, add to the MAP weight, run one
        forward pass, then restore the original weights.
        """
        if offsets is None:
            offsets = self._build_offsets(base_model, param_map)

        mean_flat = self.la_model.mean
        sample_flat = self.la_model.sample(n_samples=1)[0]
        delta_flat = sample_flat - mean_flat

        originals = {}
        named = dict(base_model.named_parameters())

        try:
            for i, name in param_map.items():
                param = named[name]
                originals[name] = param.data.clone()

                start, end = offsets[name]
                delta = delta_flat[start:end].view_as(param)

                entry = self.decomposed[i]
                if self.n_factors(i) == 2:
                    V0 = entry[0]["eigenvectors"]
                    V1 = entry[1]["eigenvectors"]
                    mask0 = entry[0]["masks"][mc_idx]
                    mask1 = entry[1]["masks"][mc_idx]

                    delta_2d = (
                        delta.reshape(delta.shape[0], -1) if delta.dim() > 2 else delta
                    )
                    out_dim, in_dim = delta_2d.shape

                    if V0.shape[0] == in_dim and V1.shape[0] == out_dim:
                        V_A, V_B = V0, V1
                        mask_A, mask_B = mask0, mask1
                    elif V0.shape[0] == out_dim and V1.shape[0] == in_dim:
                        V_A, V_B = V1, V0
                        mask_A, mask_B = mask1, mask0
                    else:
                        raise ValueError(
                            f"Factor shapes {V0.shape}, {V1.shape} don't match "
                            f"delta's dims (out={out_dim}, in={in_dim}) for '{name}' (i={i})."
                        )

                    delta_tilde = V_B.T @ delta_2d @ V_A
                    delta_tilde = (
                        delta_tilde * mask_B.unsqueeze(1) * mask_A.unsqueeze(0)
                    )
                    delta_masked = (V_B @ delta_tilde @ V_A.T).reshape(delta.shape)
                else:
                    V = entry[0]["eigenvectors"]
                    mask = entry[0]["masks"][mc_idx]
                    delta_tilde = (V.T @ delta) * mask
                    delta_masked = V @ delta_tilde

                param.data.copy_(param.data + delta_masked)

            with torch.no_grad():
                out = base_model(x)
        finally:
            for name, orig in originals.items():
                named[name].data.copy_(orig)

        return out

    def predictive_samples(self, base_model, x, param_map, mc_idices, n_samples=None):
        """
        Returns (n_samples, batch, num_classes) softmax probabilities,
        using masked_forward_via_la_sample under the hood.
        """
        n_samples = n_samples or self.M
        offsets = self._build_offsets(base_model, param_map)
        probs_T = []
        for _ in range(n_samples):
            index = mc_idices[torch.randint(0, len(mc_idices), ()).item()]
            logits = self.masked_forward_via_la_sample(
                base_model, x, param_map, index, offsets=offsets
            )
            probs_T.append(torch.softmax(logits, dim=-1).unsqueeze(0))
        return torch.cat(probs_T, dim=0)
