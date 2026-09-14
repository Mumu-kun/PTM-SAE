"""Hugging Face-style configuration classes for baseline monolithic Sparse Autoencoders."""

from transformers import PretrainedConfig


class SAEConfig(PretrainedConfig):
    """Base configuration shared by all monolithic baseline SAE variants."""

    model_type = "ptm-sae"

    def __init__(self, d_in: int = 1280, d_hidden: int = 4096, **kwargs):
        self.d_in = d_in
        self.d_hidden = d_hidden
        super().__init__(**kwargs)


class TopKSAEConfig(SAEConfig):
    """Configuration for the exact k-sparse TopK SAE (Gao et al., 2024)."""

    model_type = "ptm-sae-topk"

    def __init__(
        self,
        d_in: int = 1280,
        d_hidden: int = 4096,
        k: int = 32,
        auxk_coefficient: float = 0.0,
        k_aux: int | None = None,
        **kwargs,
    ):
        self.k = k
        # AuxK (dead-latent auxiliary reconstruction loss) is off by default, matching the
        # "track and report only" baseline decision — set auxk_coefficient > 0 to enable it as
        # an ablation parameter. k_aux defaults to a power of two near d_in/2 if left unset
        # (Gao et al., 2024's own heuristic: 512 for GPT-2 small's d_model=768).
        self.auxk_coefficient = auxk_coefficient
        self.k_aux = k_aux
        super().__init__(d_in=d_in, d_hidden=d_hidden, **kwargs)


class JumpReLUSAEConfig(SAEConfig):
    """Configuration for the step-threshold JumpReLU SAE (Rajamanoharan et al., 2024)."""

    model_type = "ptm-sae-jumprelu"

    def __init__(
        self,
        d_in: int = 1280,
        d_hidden: int = 4096,
        bandwidth: float = 1e-3,
        init_threshold: float = 0.001,
        **kwargs,
    ):
        self.bandwidth = bandwidth
        self.init_threshold = init_threshold
        super().__init__(d_in=d_in, d_hidden=d_hidden, **kwargs)


class BatchTopKSAEConfig(SAEConfig):
    """Configuration for BatchTopK: a per-batch (not per-residue) sparsity budget, letting
    complex residues borrow more latents than simple ones (Bussmann, 2024)."""

    model_type = "ptm-sae-batchtopk"

    def __init__(
        self,
        d_in: int = 1280,
        d_hidden: int = 4096,
        k: int = 32,
        auxk_coefficient: float = 1.0 / 32.0,
        k_aux: int | None = None,
        threshold_ema_decay: float = 0.99,
        **kwargs,
    ):
        self.k = k
        # AuxK is on by default here, matching BatchTopK's own published recipe (unlike plain
        # TopK, where it's an opt-in ablation).
        self.auxk_coefficient = auxk_coefficient
        self.k_aux = k_aux
        self.threshold_ema_decay = threshold_ema_decay
        super().__init__(d_in=d_in, d_hidden=d_hidden, **kwargs)


class GatedSAEConfig(SAEConfig):
    """Configuration for Gated SAE: a hard gate (which latents fire) trained separately from
    a magnitude path (how strong), avoiding L1 shrinkage without needing a straight-through
    estimator (Rajamanoharan et al., 2024a).

    No architecture-specific fields beyond d_in/d_hidden: the model returns raw `mse_loss`,
    `l1_loss`, and `aux_loss` components (mirroring how JumpReLUSAEConfig carries no
    l0_coefficient either) — the L1 penalty weight is a training-loop concern, set in
    SAETrainingConfig, not baked into the model.
    """

    model_type = "ptm-sae-gated"
