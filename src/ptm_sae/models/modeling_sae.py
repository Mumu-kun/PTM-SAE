"""Hugging Face `PreTrainedModel` wrappers for baseline monolithic Sparse Autoencoders.

Subclassing `PreTrainedModel` gives every SAE variant `save_pretrained` / `from_pretrained` /
`push_to_hub` for free, consistent with the rest of the project's Hugging Face Hub-native
storage model (`mustafa-muhaimin/ptm-sae-dataset`). Callers remain responsible for invoking
`normalize_decoder_()` after each optimizer step and, optionally,
`remove_decoder_gradient_parallel_component_()` immediately before it — these are training-loop
concerns that don't fit `PreTrainedModel`'s forward-pass contract. Loss coefficients (AuxK's
weight, JumpReLU's L0 weight, Gated SAE's L1 weight) likewise live in the training loop, not
the model: every `forward()` returns raw, unweighted components.

`SAEPreTrainedModel` is intentionally minimal — it owns only the decoder (`W_dec`, `b_dec`)
and the utilities that operate purely on it. Each concrete model defines its own encoder shape
directly, rather than sharing an intermediate tier: TopK/JumpReLU/BatchTopK all happen to use
an identical single dense encoder matrix (a few duplicated lines each), while Gated SAE's
two-path encoder never fit that shape at all.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from ptm_sae.models.configuration_sae import (
    BatchTopKSAEConfig,
    GatedSAEConfig,
    JumpReLUSAEConfig,
    SAEConfig,
    TopKSAEConfig,
)


def _geometric_median(
    points: torch.Tensor, n_iters: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    """Weiszfeld's algorithm: the point minimizing summed L2 distance to `points` (shape
    (n, d)). Unlike the coordinate-wise mean, it is robust to outlier rows — each point's
    pull on the next estimate is weighted by 1/distance, so far-away points get down-weighted
    as the estimate converges toward the dense cluster."""
    median = points.mean(dim=0)
    for _ in range(n_iters):
        distances = (points - median).norm(dim=-1).clamp_min(eps)
        weights = 1.0 / distances
        median = (weights.unsqueeze(-1) * points).sum(dim=0) / weights.sum()
    return median


def _nearest_power_of_two(x: int) -> int:
    """Nearest power of two to x — Gao et al. (2024)'s own heuristic for AuxK's k_aux
    (they use 512, a power of two close to GPT-2 small's d_model/2 = 384)."""
    if x <= 1:
        return 1
    lower = 1 << (x.bit_length() - 1)
    upper = lower << 1
    return lower if (x - lower) < (upper - x) else upper


@dataclass
class SAEOutput(ModelOutput):
    """Forward-pass result: reconstruction, sparse latent code, and raw (unweighted) loss
    components. `l1_loss` (Gated SAE) and `aux_loss` (AuxK, any architecture that supports it)
    are None unless the architecture/config actually produces them."""

    reconstruction: torch.Tensor = None
    latents: torch.Tensor = None
    mse_loss: torch.Tensor | None = None
    l0: torch.Tensor | None = None
    l1_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


class SAEPreTrainedModel(PreTrainedModel):
    """Minimal shared base: decoder, pre-encoder bias, and decoder-constraint utilities only."""

    config_class = SAEConfig
    base_model_prefix = "sae"
    main_input_name = "activations"

    def __init__(self, config: SAEConfig):
        super().__init__(config)

        self.b_dec = nn.Parameter(torch.zeros(config.d_in))

        w_dec = torch.randn(config.d_hidden, config.d_in)
        w_dec = w_dec / w_dec.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)

    def _init_weights(self, module):
        """No-op: each concrete model sets its own tied initialization explicitly in __init__."""

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents @ self.W_dec + self.b_dec

    def reconstruction_loss(
        self, activations: torch.Tensor, reconstruction: torch.Tensor
    ) -> torch.Tensor:
        return (reconstruction - activations).pow(2).mean()

    @torch.no_grad()
    def initialize_bias_from_data(self, sample_batch: torch.Tensor) -> None:
        """Seeds `b_dec` from the geometric median of `sample_batch` (shape (n_tokens, d_in)),
        instead of leaving it at zero. Standard SAE practice (Anthropic's *Towards
        Monosemanticity*; Gao et al., 2024) — call once, right after construction and before
        the optimizer exists, on a throwaway sample of real activations."""
        self.b_dec.copy_(_geometric_median(sample_batch.to(self.b_dec.device)))

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

    def auxk_loss(
        self,
        pre_acts: torch.Tensor,
        residual: torch.Tensor,
        dead_latent_mask: torch.Tensor,
        k_aux: int,
    ) -> torch.Tensor:
        """AuxK (Gao et al., 2024): reconstructs the main path's (detached) residual using
        only the top-k_aux latents among currently-dead ones, giving them a gradient signal
        even though hard top-k selection zeroes their gradient through the main path entirely.
        Shared by TopK/BatchTopK — the two architectures whose hard selection can do this.

        `dead_latent_mask` (shape (d_hidden,), bool) comes from the training loop's own
        token-windowed census — the model stays stateless, so it doesn't track this itself.
        """
        n_dead = int(dead_latent_mask.sum().item())
        if n_dead == 0:
            return torch.zeros((), device=pre_acts.device, dtype=pre_acts.dtype)
        k_eff = min(k_aux, n_dead)

        relu_pre_acts = torch.relu(pre_acts)
        # Push non-dead latents below any valid (>=0) activation so they can never be selected.
        masked = relu_pre_acts.masked_fill(~dead_latent_mask, -1.0)
        topk_vals, topk_idx = torch.topk(masked, k_eff, dim=-1)
        z_aux = torch.zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals.clamp_min(0))

        reconstruction_aux = z_aux @ self.W_dec  # no b_dec: predicting a residual, not x itself
        return (residual.detach() - reconstruction_aux).pow(2).mean()


class TopKSAEModel(SAEPreTrainedModel):
    """Monolithic TopK SAE: exact k-sparse latent code via hard top-k selection (Gao et al., 2024)."""

    config_class = TopKSAEConfig

    def __init__(self, config: TopKSAEConfig):
        super().__init__(config)
        if not 0 < config.k <= config.d_hidden:
            raise ValueError(f"k ({config.k}) must be in (0, d_hidden={config.d_hidden}]")
        self.k = config.k
        self.k_aux = (
            config.k_aux if config.k_aux is not None else _nearest_power_of_two(config.d_in // 2)
        )

        self.b_enc = nn.Parameter(torch.zeros(config.d_hidden))
        self.W_enc = nn.Parameter(self.W_dec.t().clone())
        self.post_init()

    def encode_pre_activation(self, activations: torch.Tensor) -> torch.Tensor:
        return (activations - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        pre_acts = torch.relu(self.encode_pre_activation(activations))
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        return torch.zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals)

    def forward(
        self, activations: torch.Tensor, dead_latent_mask: torch.Tensor | None = None
    ) -> SAEOutput:
        latents = self.encode(activations)
        reconstruction = self.decode(latents)
        mse_loss = self.reconstruction_loss(activations, reconstruction)
        l0 = latents.gt(0).sum(dim=-1).float().mean()

        # AuxK only ever shapes training gradients — under eval (no backward pass at all) it
        # would just be wasted compute with no purpose, so it's skipped entirely rather than
        # forcing every eval call to supply a meaningless mask.
        aux_loss = None
        if self.config.auxk_coefficient > 0 and self.training:
            if dead_latent_mask is None:
                raise ValueError(
                    "auxk_coefficient > 0 requires dead_latent_mask (from the training loop's "
                    "token-windowed dead-latent census) to be passed to forward() while training."
                )
            pre_acts = self.encode_pre_activation(activations)
            residual = activations - reconstruction
            aux_loss = self.auxk_loss(pre_acts, residual, dead_latent_mask, self.k_aux)

        return SAEOutput(
            reconstruction=reconstruction,
            latents=latents,
            mse_loss=mse_loss,
            l0=l0,
            aux_loss=aux_loss,
        )


class BatchTopKSAEModel(SAEPreTrainedModel):
    """BatchTopK: relaxes TopK's fixed per-residue k to a per-batch budget (Bussmann, 2024) —
    some residues borrow more latents than others within a batch, while the batch average
    matches k. Falls back to a running-average threshold (tracked only during training) at
    eval time, since there's no batch to pool over for a single-example inference call.
    """

    config_class = BatchTopKSAEConfig
    running_threshold: torch.Tensor
    threshold_initialized: torch.Tensor

    def __init__(self, config: BatchTopKSAEConfig):
        super().__init__(config)
        if not 0 < config.k <= config.d_hidden:
            raise ValueError(f"k ({config.k}) must be in (0, d_hidden={config.d_hidden}]")
        self.k = config.k
        self.k_aux = (
            config.k_aux if config.k_aux is not None else _nearest_power_of_two(config.d_in // 2)
        )
        self.threshold_ema_decay = config.threshold_ema_decay

        self.b_enc = nn.Parameter(torch.zeros(config.d_hidden))
        self.W_enc = nn.Parameter(self.W_dec.t().clone())

        # Running estimate of theta = E_X[min positive activation per example], used only at
        # eval time (no batch to pool a joint top-k over for a single inference call). Updated
        # only during training; a buffer, not a parameter — never touched by the optimizer.
        self.register_buffer("running_threshold", torch.tensor(0.0))
        self.register_buffer("threshold_initialized", torch.tensor(False))
        self.post_init()

    def encode_pre_activation(self, activations: torch.Tensor) -> torch.Tensor:
        return (activations - self.b_dec) @ self.W_enc + self.b_enc

    def _select_batch_topk(self, relu_pre_acts: torch.Tensor) -> torch.Tensor:
        batch_size = relu_pre_acts.shape[0]
        budget = min(batch_size * self.k, relu_pre_acts.numel())
        flat = relu_pre_acts.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat, budget)
        return torch.zeros_like(flat).scatter_(-1, topk_idx, topk_vals).reshape_as(relu_pre_acts)

    @torch.no_grad()
    def _update_running_threshold(self, latents: torch.Tensor) -> None:
        positive = latents.masked_fill(latents <= 0, float("inf"))
        row_min = positive.min(dim=-1).values
        valid = torch.isfinite(row_min)
        if not valid.any():
            return
        batch_estimate = row_min[valid].mean()
        if not bool(self.threshold_initialized):
            self.running_threshold.copy_(batch_estimate)
            self.threshold_initialized.fill_(True)
        else:
            decay = self.threshold_ema_decay
            self.running_threshold.mul_(decay).add_(batch_estimate, alpha=1.0 - decay)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encode_pre_activation(activations)
        relu_pre_acts = torch.relu(pre_acts)
        if self.training:
            latents = self._select_batch_topk(relu_pre_acts)
            self._update_running_threshold(latents)
            return latents
        # Eval time: no batch to pool a joint top-k over, so fall back to the running threshold
        # (JumpReLU-style fixed cutoff). Before any training step has run, this is 0 — i.e. no
        # sparsity constraint yet, an expected cold-start rather than an error.
        return relu_pre_acts * (relu_pre_acts > self.running_threshold).to(relu_pre_acts.dtype)

    def forward(
        self, activations: torch.Tensor, dead_latent_mask: torch.Tensor | None = None
    ) -> SAEOutput:
        latents = self.encode(activations)
        reconstruction = self.decode(latents)
        mse_loss = self.reconstruction_loss(activations, reconstruction)
        l0 = latents.gt(0).sum(dim=-1).float().mean()

        # AuxK only ever shapes training gradients — see TopKSAEModel.forward for why eval
        # skips it entirely rather than requiring a meaningless mask.
        aux_loss = None
        if self.config.auxk_coefficient > 0 and self.training:
            if dead_latent_mask is None:
                raise ValueError(
                    "auxk_coefficient > 0 requires dead_latent_mask (from the training loop's "
                    "token-windowed dead-latent census) to be passed to forward() while training."
                )
            pre_acts = self.encode_pre_activation(activations)
            residual = activations - reconstruction
            aux_loss = self.auxk_loss(pre_acts, residual, dead_latent_mask, self.k_aux)

        return SAEOutput(
            reconstruction=reconstruction,
            latents=latents,
            mse_loss=mse_loss,
            l0=l0,
            aux_loss=aux_loss,
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

        self.b_enc = nn.Parameter(torch.zeros(config.d_hidden))
        self.W_enc = nn.Parameter(self.W_dec.t().clone())
        self.post_init()

    def encode_pre_activation(self, activations: torch.Tensor) -> torch.Tensor:
        return (activations - self.b_dec) @ self.W_enc + self.b_enc

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

        return SAEOutput(reconstruction=reconstruction, latents=latents, mse_loss=mse_loss, l0=l0)


class GatedSAEModel(SAEPreTrainedModel):
    """Gated SAE (Rajamanoharan et al., 2024a): a hard gate decides which latents fire, a
    separate magnitude path (weight-tied to the gate via a per-latent rescale exp(r_mag))
    estimates how strong each firing latent should be. No straight-through estimator needed —
    the gate is trained via a frozen-decoder auxiliary reconstruction plus an L1 penalty on the
    gate's own rectified pre-activations, never through the hard gate itself. The magnitude
    path shares W_gate's directions and gets ordinary reconstruction-loss gradient directly,
    which is what keeps W_gate's directions trained even when a given latent's gate rarely
    fires — unlike TopK, there's no route through which a latent's encoder weights are
    guaranteed to receive exactly zero gradient.
    """

    config_class = GatedSAEConfig

    def __init__(self, config: GatedSAEConfig):
        super().__init__(config)
        self.b_gate = nn.Parameter(torch.zeros(config.d_hidden))
        self.b_mag = nn.Parameter(torch.zeros(config.d_hidden))
        self.r_mag = nn.Parameter(torch.zeros(config.d_hidden))  # exp(0) = 1: W_mag starts = W_gate
        self.W_gate = nn.Parameter(self.W_dec.t().clone())
        self.post_init()

    def gate_pre_activation(self, activations: torch.Tensor) -> torch.Tensor:
        return (activations - self.b_dec) @ self.W_gate + self.b_gate

    def magnitude_pre_activation(self, activations: torch.Tensor) -> torch.Tensor:
        w_mag = self.W_gate * torch.exp(self.r_mag)  # per-latent rescale, broadcasts on d_hidden
        return (activations - self.b_dec) @ w_mag + self.b_mag

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        gate = (self.gate_pre_activation(activations) > 0).to(activations.dtype)
        magnitude = torch.relu(self.magnitude_pre_activation(activations))
        return gate * magnitude

    def forward(self, activations: torch.Tensor) -> SAEOutput:
        pi_gate = self.gate_pre_activation(activations)
        gate = (pi_gate > 0).to(activations.dtype)
        magnitude = torch.relu(self.magnitude_pre_activation(activations))
        latents = gate * magnitude

        reconstruction = self.decode(latents)
        mse_loss = self.reconstruction_loss(activations, reconstruction)
        l0 = latents.gt(0).sum(dim=-1).float().mean()

        relu_gate = torch.relu(pi_gate)
        l1_loss = relu_gate.abs().sum(dim=-1).mean()

        # Frozen-decoder auxiliary reconstruction (Eq. 8): trains the gate path via a real
        # reconstruction target without letting gradient reach W_dec through this side term.
        # b_dec is intentionally NOT fully isolated: relu_gate already depends on it through
        # gate_pre_activation's (non-detached) "activations - b_dec" centering step, so b_dec's
        # encoder-centering role still receives gradient here, same as W_gate/b_gate do — only
        # its decoder-output use (the "+ b_dec" below) is stopped.
        frozen_reconstruction = relu_gate @ self.W_dec.detach() + self.b_dec.detach()
        aux_loss = self.reconstruction_loss(activations, frozen_reconstruction)

        return SAEOutput(
            reconstruction=reconstruction,
            latents=latents,
            mse_loss=mse_loss,
            l0=l0,
            l1_loss=l1_loss,
            aux_loss=aux_loss,
        )
