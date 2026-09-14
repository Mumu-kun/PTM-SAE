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
        self, d_in: int = 1280, d_hidden: int = 4096, k: int = 32, **kwargs
    ):
        self.k = k
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
