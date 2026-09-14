"""Hugging Face `PreTrainedModel` wrappers for baseline monolithic Sparse Autoencoders.

Subclassing `PreTrainedModel` gives every SAE variant `save_pretrained` / `from_pretrained` /
`push_to_hub` for free, consistent with the rest of the project's Hugging Face Hub-native
storage model (`mustafa-muhaimin/ptm-sae-dataset`). Callers remain responsible for invoking
`normalize_decoder_()` after each optimizer step and, optionally,
`remove_decoder_gradient_parallel_component_()` immediately before it — these are training-loop
concerns that don't fit `PreTrainedModel`'s forward-pass contract.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from ptm_sae.models.configuration_sae import JumpReLUSAEConfig, SAEConfig, TopKSAEConfig


@dataclass
class SAEOutput(ModelOutput):
    """Forward-pass result: reconstruction, sparse latent code, and loss terms."""

    reconstruction: torch.Tensor = None
    latents: torch.Tensor = None
    mse_loss: torch.Tensor | None = None
    l0: torch.Tensor | None = None


class SAEPreTrainedModel(PreTrainedModel):
    """Shared pre-bias, encoder/decoder linear maps, and unit-norm decoder constraint."""

    config_class = SAEConfig
    base_model_prefix = "sae"
    main_input_name = "activations"

    def __init__(self, config: SAEConfig):
        super().__init__(config)

        self.b_dec = nn.Parameter(torch.zeros(config.d_in))
        self.b_enc = nn.Parameter(torch.zeros(config.d_hidden))

        # Tied initialization: decoder directions unit-norm, encoder starts as its transpose.
        w_dec = torch.randn(config.d_hidden, config.d_in)
        w_dec = w_dec / w_dec.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)
        self.W_enc = nn.Parameter(w_dec.t().clone())

    def _init_weights(self, module):
        """No-op: the tied encoder/decoder initialization above is authoritative."""

    def encode_pre_activation(self, activations: torch.Tensor) -> torch.Tensor:
        return (activations - self.b_dec) @ self.W_enc + self.b_enc

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents @ self.W_dec + self.b_dec

    def reconstruction_loss(
        self, activations: torch.Tensor, reconstruction: torch.Tensor
    ) -> torch.Tensor:
        return (reconstruction - activations).pow(2).mean()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Renormalizes decoder feature directions back to unit norm. Call after each optimizer step."""
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def remove_decoder_gradient_parallel_component_(self) -> None:
        """Projects out the decoder gradient component parallel to each feature direction.

        Prevents gradient descent from fighting the unit-norm renormalization applied after
        the optimizer step. Call after `loss.backward()` and before `optimizer.step()`.
        """
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec).sum(dim=-1, keepdim=True) * self.W_dec
        self.W_dec.grad -= parallel


class TopKSAEModel(SAEPreTrainedModel):
    """Monolithic TopK SAE: exact k-sparse latent code via hard top-k selection (Gao et al., 2024)."""

    config_class = TopKSAEConfig

    def __init__(self, config: TopKSAEConfig):
        super().__init__(config)
        if not 0 < config.k <= config.d_hidden:
            raise ValueError(f"k ({config.k}) must be in (0, d_hidden={config.d_hidden}]")
        self.k = config.k
        self.post_init()

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        pre_acts = torch.relu(self.encode_pre_activation(activations))
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        return torch.zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals)

    def forward(self, activations: torch.Tensor) -> SAEOutput:
        latents = self.encode(activations)
        reconstruction = self.decode(latents)
        mse_loss = self.reconstruction_loss(activations, reconstruction)
        l0 = latents.gt(0).sum(dim=-1).float().mean()
        return SAEOutput(
            reconstruction=reconstruction, latents=latents, mse_loss=mse_loss, l0=l0
        )


class _JumpReLUSTE(torch.autograd.Function):
    """Heaviside-gated activation z = x * 1[x > threshold] with a rectangular-kernel
    pseudo-derivative for the (otherwise zero-everywhere) threshold gradient."""

    @staticmethod
    def forward(ctx, pre_acts: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float):
        threshold = torch.exp(log_threshold)
        mask = (pre_acts > threshold).to(pre_acts.dtype)
        ctx.save_for_backward(pre_acts, threshold)
        ctx.bandwidth = bandwidth
        return pre_acts * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_acts, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        mask = (pre_acts > threshold).to(grad_output.dtype)
        grad_pre_acts = grad_output * mask

        # d(JumpReLU)/d(theta) = -(theta / bandwidth) * kernel (Rajamanoharan et al., 2024, Eq. 11).
        # log_threshold is the trained parameter, so the chain rule needs one more factor of
        # theta: d/d(log theta) = d/d(theta) * d(theta)/d(log theta) = d/d(theta) * theta.
        kernel = ((pre_acts - threshold).abs() < (bandwidth / 2.0)).to(grad_output.dtype)
        grad_threshold = -(threshold / bandwidth) * kernel * grad_output
        grad_log_threshold = (grad_threshold * threshold).sum(dim=0)

        return grad_pre_acts, grad_log_threshold, None


class _HeavisideSTE(torch.autograd.Function):
    """Differentiable active-latent indicator 1[x > threshold] used for the L0 pseudo-count.

    Carries no gradient to `pre_acts` (the count itself is locally constant almost everywhere);
    only the threshold receives a pseudo-derivative via the same rectangular kernel.
    """

    @staticmethod
    def forward(ctx, pre_acts: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float):
        threshold = torch.exp(log_threshold)
        ctx.save_for_backward(pre_acts, threshold)
        ctx.bandwidth = bandwidth
        return (pre_acts > threshold).to(pre_acts.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_acts, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # d(Heaviside)/d(theta) = -(1 / bandwidth) * kernel (Rajamanoharan et al., 2024, Eq. 12),
        # then the same log-space chain-rule factor of theta as _JumpReLUSTE above.
        kernel = ((pre_acts - threshold).abs() < (bandwidth / 2.0)).to(grad_output.dtype)
        grad_threshold = -(1.0 / bandwidth) * kernel * grad_output
        grad_log_threshold = (grad_threshold * threshold).sum(dim=0)

        return torch.zeros_like(pre_acts), grad_log_threshold, None


class JumpReLUSAEModel(SAEPreTrainedModel):
    """Step-threshold JumpReLU SAE with straight-through pseudo-derivatives
    (Rajamanoharan et al., 2024). Per-latent threshold is trained in log-space to stay positive.
    """

    config_class = JumpReLUSAEConfig

    def __init__(self, config: JumpReLUSAEConfig):
        super().__init__(config)
        self.bandwidth = config.bandwidth
        self.log_threshold = nn.Parameter(
            torch.full((config.d_hidden,), math.log(config.init_threshold))
        )
        self.post_init()

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encode_pre_activation(activations)
        return _JumpReLUSTE.apply(pre_acts, self.log_threshold, self.bandwidth)

    def forward(self, activations: torch.Tensor) -> SAEOutput:
        pre_acts = self.encode_pre_activation(activations)
        latents = _JumpReLUSTE.apply(pre_acts, self.log_threshold, self.bandwidth)
        reconstruction = self.decode(latents)
        mse_loss = self.reconstruction_loss(activations, reconstruction)

        active = _HeavisideSTE.apply(pre_acts, self.log_threshold, self.bandwidth)
        l0 = active.sum(dim=-1).mean()

        return SAEOutput(
            reconstruction=reconstruction, latents=latents, mse_loss=mse_loss, l0=l0
        )
