"""Test suite for baseline TopK and JumpReLU SAE architectures (transformers.PreTrainedModel)."""

import math

import torch

from ptm_sae.models import (
    BatchTopKSAEConfig,
    BatchTopKSAEModel,
    GatedSAEConfig,
    GatedSAEModel,
    JumpReLUSAEConfig,
    JumpReLUSAEModel,
    TopKSAEConfig,
    TopKSAEModel,
)
from ptm_sae.models.modeling_sae import (
    _HeavisideSTE,
    _JumpReLUSTE,
    _nearest_power_of_two,
)


def test_topk_sae_exact_sparsity_and_shapes():
    """Verify TopKSAEModel enforces exactly k nonzero latents per row and reconstructs input shape."""
    torch.manual_seed(0)
    d_in, d_hidden, k, batch = 16, 64, 8, 32
    sae = TopKSAEModel(TopKSAEConfig(d_in=d_in, d_hidden=d_hidden, k=k))

    x = torch.randn(batch, d_in)
    out = sae(x)

    assert out.reconstruction.shape == (batch, d_in)
    assert out.latents.shape == (batch, d_hidden)
    assert torch.all(out.latents.gt(0).sum(dim=-1) <= k)
    assert out.l0.item() <= k


def test_topk_sae_rejects_invalid_k():
    """k must be a positive integer not exceeding dictionary width."""
    try:
        TopKSAEModel(TopKSAEConfig(d_in=8, d_hidden=16, k=0))
        raise AssertionError("expected ValueError for k=0")
    except ValueError:
        pass

    try:
        TopKSAEModel(TopKSAEConfig(d_in=8, d_hidden=16, k=32))
        raise AssertionError("expected ValueError for k > d_hidden")
    except ValueError:
        pass


def test_topk_sae_gradients_flow():
    """Verify backward pass populates gradients on encoder/decoder/bias parameters."""
    torch.manual_seed(1)
    sae = TopKSAEModel(TopKSAEConfig(d_in=8, d_hidden=32, k=4))
    x = torch.randn(16, 8)

    out = sae(x)
    out.mse_loss.backward()

    assert sae.W_enc.grad is not None
    assert sae.W_dec.grad is not None
    assert sae.b_dec.grad is not None
    assert sae.b_enc.grad is not None


def test_decoder_normalization_and_gradient_projection():
    """Verify unit-norm renormalization and parallel-gradient-component removal."""
    torch.manual_seed(2)
    sae = TopKSAEModel(TopKSAEConfig(d_in=8, d_hidden=16, k=4))

    # Decoder starts unit-norm by construction; perturb and renormalize.
    with torch.no_grad():
        sae.W_dec.mul_(3.0)
    sae.normalize_decoder_()
    norms = sae.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    x = torch.randn(8, 8)
    out = sae(x)
    out.mse_loss.backward()
    sae.remove_decoder_gradient_parallel_component_()

    parallel_component = (sae.W_dec.grad * sae.W_dec).sum(dim=-1)
    assert torch.allclose(
        parallel_component, torch.zeros_like(parallel_component), atol=1e-5
    )


def test_topk_sae_save_and_load_round_trip(tmp_path):
    """Verify save_pretrained/from_pretrained preserves config and learned parameters exactly."""
    torch.manual_seed(6)
    sae = TopKSAEModel(TopKSAEConfig(d_in=8, d_hidden=16, k=4))
    sae.save_pretrained(tmp_path)

    reloaded = TopKSAEModel.from_pretrained(tmp_path)
    assert reloaded.config.d_in == 8
    assert reloaded.config.d_hidden == 16
    assert reloaded.k == 4
    assert torch.equal(reloaded.W_dec, sae.W_dec)
    assert torch.equal(reloaded.W_enc, sae.W_enc)

    x = torch.randn(4, 8)
    torch.manual_seed(0)
    original_out = sae(x)
    torch.manual_seed(0)
    reloaded_out = reloaded(x)
    assert torch.equal(original_out.reconstruction, reloaded_out.reconstruction)


def test_jumprelu_sae_forward_and_shapes():
    """Verify JumpReLUSAEModel forward pass shapes and non-negative sparsity count."""
    torch.manual_seed(3)
    d_in, d_hidden, batch = 16, 64, 32
    sae = JumpReLUSAEModel(JumpReLUSAEConfig(d_in=d_in, d_hidden=d_hidden))

    x = torch.randn(batch, d_in)
    out = sae(x)

    assert out.reconstruction.shape == (batch, d_in)
    assert out.latents.shape == (batch, d_hidden)
    assert out.l0.item() >= 0.0


def test_jumprelu_sae_threshold_gradient_flows():
    """Verify the straight-through estimator routes gradient into log_threshold via both
    the reconstruction loss and the L0 pseudo-count term."""
    torch.manual_seed(4)
    sae = JumpReLUSAEModel(
        JumpReLUSAEConfig(d_in=8, d_hidden=32, bandwidth=1e-2, init_threshold=0.01)
    )
    x = torch.randn(64, 8)

    out = sae(x)
    total_loss = out.mse_loss + 1e-3 * out.l0
    total_loss.backward()

    assert sae.log_threshold.grad is not None
    assert torch.any(sae.log_threshold.grad != 0)


def test_jumprelu_sae_masks_below_threshold():
    """Latents strictly below the learned threshold must be exactly zero."""
    torch.manual_seed(5)
    sae = JumpReLUSAEModel(JumpReLUSAEConfig(d_in=8, d_hidden=32, init_threshold=0.5))
    x = torch.randn(64, 8) * 0.1  # small activations, mostly below threshold

    with torch.no_grad():
        pre_acts = sae.encode_pre_activation(x)
        latents = sae.encode(x)
        threshold = torch.exp(sae.log_threshold)

    below = pre_acts <= threshold
    assert torch.all(latents[below] == 0)


def test_jumprelu_ste_threshold_gradient_matches_closed_form():
    """Regression test for a scaling bug: log_threshold's gradient must include the log-space
    chain-rule factor of theta (d(theta)/d(log theta) = theta) on top of the paper's raw-theta
    pseudo-derivative (Rajamanoharan et al., 2024, Eq. 11/12) — not just the raw-theta formula
    alone, which under-scaled the reconstruction-path gradient by 1/theta relative to the
    correctly-scaled L0-path gradient.
    """
    bandwidth = 0.1
    theta = 0.5

    log_theta = torch.tensor([math.log(theta)], requires_grad=True)
    pre_acts = torch.tensor([[theta]])  # sits exactly on the threshold: inside the kernel window
    out = _JumpReLUSTE.apply(pre_acts, log_theta, bandwidth)
    out.sum().backward()
    expected_value_grad = -(theta**2 / bandwidth)  # -(theta/bandwidth)*kernel(=1) * theta
    assert torch.allclose(log_theta.grad, torch.tensor([expected_value_grad]), atol=1e-6)

    log_theta_l0 = torch.tensor([math.log(theta)], requires_grad=True)
    active = _HeavisideSTE.apply(pre_acts, log_theta_l0, bandwidth)
    active.sum().backward()
    expected_l0_grad = -(theta / bandwidth)  # -(1/bandwidth)*kernel(=1) * theta
    assert torch.allclose(log_theta_l0.grad, torch.tensor([expected_l0_grad]), atol=1e-6)


def test_jumprelu_sae_save_and_load_round_trip(tmp_path):
    """Verify save_pretrained/from_pretrained preserves the learned per-latent thresholds."""
    torch.manual_seed(7)
    sae = JumpReLUSAEModel(JumpReLUSAEConfig(d_in=8, d_hidden=16))
    sae.save_pretrained(tmp_path)

    reloaded = JumpReLUSAEModel.from_pretrained(tmp_path)
    assert reloaded.config.bandwidth == sae.config.bandwidth
    assert torch.equal(reloaded.log_threshold, sae.log_threshold)


def test_nearest_power_of_two():
    """Verify AuxK's k_aux default heuristic (Gao et al., 2024: power of two near d_model/2)."""
    assert _nearest_power_of_two(640) == 512  # our d_in=1280 case: 640 is closer to 512 than 1024
    assert _nearest_power_of_two(384) == 512  # GPT-2 small's own d_model/2 case from the paper
    assert _nearest_power_of_two(1) == 1


def test_topk_auxk_disabled_by_default_no_mask_needed():
    """Verify plain TopK (auxk_coefficient=0.0 default) never requires a dead_latent_mask."""
    torch.manual_seed(10)
    sae = TopKSAEModel(TopKSAEConfig(d_in=8, d_hidden=32, k=4))
    x = torch.randn(16, 8)

    out = sae(x)  # no dead_latent_mask passed

    assert out.aux_loss is None


def test_topk_auxk_enabled_requires_mask():
    """Verify auxk_coefficient > 0 without a dead_latent_mask raises, rather than silently
    skipping AuxK (the model has no state of its own to infer dead latents from)."""
    torch.manual_seed(11)
    sae = TopKSAEModel(TopKSAEConfig(d_in=8, d_hidden=32, k=4, auxk_coefficient=0.1))
    x = torch.randn(16, 8)

    try:
        sae(x)
        raise AssertionError("expected ValueError for missing dead_latent_mask")
    except ValueError:
        pass


def test_topk_auxk_skipped_in_eval_mode_without_mask():
    """AuxK only ever shapes training gradients; eval mode must never require a mask, since
    _evaluate() (no backward pass at all) has no meaningful use for it."""
    torch.manual_seed(21)
    sae = TopKSAEModel(TopKSAEConfig(d_in=8, d_hidden=32, k=4, auxk_coefficient=0.1))
    sae.eval()
    x = torch.randn(16, 8)

    out = sae(x)  # no dead_latent_mask, no error, because eval mode skips AuxK entirely

    assert out.aux_loss is None


def test_topk_auxk_trains_latents_dead_to_the_main_path():
    """The core AuxK claim: a latent excluded from every example's top-k gets exactly zero
    gradient from the main path alone, but a nonzero gradient once AuxK is enabled and that
    latent is marked dead."""
    torch.manual_seed(12)
    # A wide dictionary against a tiny batch/k guarantees dead latents: at most batch*k = 8
    # distinct latents can ever appear in any example's top-k, out of 64 total.
    d_in, d_hidden, k, batch = 8, 64, 2, 4
    sae = TopKSAEModel(TopKSAEConfig(d_in=d_in, d_hidden=d_hidden, k=k, auxk_coefficient=0.5))
    x = torch.randn(batch, d_in)

    # Mark every latent dead except those that actually appear in the main top-k selection —
    # guarantees at least one genuinely-dead latent whose main-path gradient is exactly zero.
    with torch.no_grad():
        latents = sae.encode(x)
        ever_fired = latents.gt(0).any(dim=0)
    dead_latent_mask = ~ever_fired
    assert dead_latent_mask.any(), "test fixture needs at least one dead latent to be meaningful"

    out = sae(x, dead_latent_mask=dead_latent_mask)
    assert out.aux_loss is not None

    out.mse_loss.backward(retain_graph=True)
    main_path_grad = sae.W_enc.grad.clone()
    assert torch.all(main_path_grad[:, dead_latent_mask] == 0)

    sae.zero_grad()
    out.aux_loss.backward()
    assert torch.any(sae.W_enc.grad[:, dead_latent_mask] != 0)


def test_batchtopk_train_mode_batch_level_sparsity():
    """Verify BatchTopK keeps exactly batch_size*k active latents total in train mode, while
    individual rows may have more or fewer than k (the whole point of relaxing to batch-level)."""
    torch.manual_seed(13)
    batch, k = 32, 4
    sae = BatchTopKSAEModel(BatchTopKSAEConfig(d_in=8, d_hidden=64, k=k, auxk_coefficient=0.0))
    sae.train()
    x = torch.randn(batch, 8)

    out = sae(x)

    assert out.latents.gt(0).sum().item() == batch * k
    per_row_counts = out.latents.gt(0).sum(dim=-1)
    assert per_row_counts.max().item() != per_row_counts.min().item()  # some real variability


def test_batchtopk_cold_start_eval_before_training():
    """Before any train-mode forward pass, running_threshold is 0 — eval mode should behave
    like plain ReLU (no sparsity constraint yet) rather than erroring."""
    torch.manual_seed(14)
    sae = BatchTopKSAEModel(BatchTopKSAEConfig(d_in=8, d_hidden=32, k=4, auxk_coefficient=0.0))
    sae.eval()
    x = torch.randn(8, 8)

    assert not bool(sae.threshold_initialized)
    out = sae(x)
    with torch.no_grad():
        expected = torch.relu(sae.encode_pre_activation(x))
    assert torch.equal(out.latents, expected)


def test_batchtopk_eval_mode_uses_running_threshold_after_training():
    """After training-mode forward passes populate running_threshold, eval mode should apply
    a real (nonzero, learned) fixed cutoff instead of the cold-start pass-through."""
    torch.manual_seed(15)
    sae = BatchTopKSAEModel(BatchTopKSAEConfig(d_in=8, d_hidden=32, k=4, auxk_coefficient=0.0))
    sae.train()
    for _ in range(5):
        sae(torch.randn(16, 8))

    assert bool(sae.threshold_initialized)
    assert sae.running_threshold.item() > 0

    sae.eval()
    x = torch.randn(8, 8)
    out = sae(x)
    with torch.no_grad():
        relu_pre_acts = torch.relu(sae.encode_pre_activation(x))
        expected = relu_pre_acts * (relu_pre_acts > sae.running_threshold).to(relu_pre_acts.dtype)
    assert torch.equal(out.latents, expected)


def test_batchtopk_auxk_enabled_by_default():
    """Verify BatchTopK matches its own published recipe: AuxK on by default (unlike plain TopK)."""
    config = BatchTopKSAEConfig(d_in=8, d_hidden=32, k=4)
    assert config.auxk_coefficient > 0


def test_batchtopk_save_load_round_trip_persists_buffers(tmp_path):
    """Verify save_pretrained/from_pretrained persists running_threshold and
    threshold_initialized — buffers, not parameters, so this is new territory versus the
    plain-parameter round-trip tests above."""
    torch.manual_seed(16)
    sae = BatchTopKSAEModel(BatchTopKSAEConfig(d_in=8, d_hidden=16, k=4, auxk_coefficient=0.0))
    sae.train()
    for _ in range(5):
        sae(torch.randn(16, 8))
    assert bool(sae.threshold_initialized)

    sae.save_pretrained(tmp_path)
    reloaded = BatchTopKSAEModel.from_pretrained(tmp_path)

    assert bool(reloaded.threshold_initialized)
    assert torch.equal(reloaded.running_threshold, sae.running_threshold)


def test_gated_sae_config_needs_no_extra_fields():
    """Verify GatedSAEConfig carries no loss-coefficient fields — those are training-loop
    concerns (SAETrainingConfig), matching how JumpReLUSAEConfig carries no l0_coefficient."""
    config = GatedSAEConfig(d_in=8, d_hidden=32)
    assert config.d_in == 8
    assert config.d_hidden == 32


def test_gated_sae_forward_shapes_and_losses():
    """Verify GatedSAEModel produces all three raw loss components (mse, l1, aux)."""
    torch.manual_seed(17)
    d_in, d_hidden, batch = 8, 32, 16
    sae = GatedSAEModel(GatedSAEConfig(d_in=d_in, d_hidden=d_hidden))
    x = torch.randn(batch, d_in)

    out = sae(x)

    assert out.reconstruction.shape == (batch, d_in)
    assert out.latents.shape == (batch, d_hidden)
    assert out.l1_loss is not None and out.l1_loss.item() >= 0
    assert out.aux_loss is not None and out.aux_loss.item() >= 0


def test_gated_sae_aux_loss_does_not_reach_real_decoder():
    """Verify the frozen-decoder auxiliary term's stop-gradient actually works: backpropagating
    aux_loss alone must leave W_dec untouched, while still training the gate path. b_dec is
    deliberately excluded from this check — it plays a dual role (subtracted before encoding,
    added after decoding), and only its decoder-output use is meant to be frozen here; its
    encoder-centering use legitimately receives gradient from the gate path, same as W_gate/
    b_gate do."""
    torch.manual_seed(18)
    sae = GatedSAEModel(GatedSAEConfig(d_in=8, d_hidden=16))
    x = torch.randn(16, 8)

    out = sae(x)
    out.aux_loss.backward()

    assert sae.W_dec.grad is None
    assert sae.W_gate.grad is not None
    assert torch.any(sae.W_gate.grad != 0)


def test_gated_sae_main_reconstruction_trains_real_decoder():
    """Sanity check the main path is unaffected by the aux term's isolation: mse_loss alone
    must still train W_dec normally."""
    torch.manual_seed(19)
    sae = GatedSAEModel(GatedSAEConfig(d_in=8, d_hidden=16))
    x = torch.randn(16, 8)

    out = sae(x)
    out.mse_loss.backward()

    assert sae.W_dec.grad is not None
    assert torch.any(sae.W_dec.grad != 0)


def test_gated_sae_save_load_round_trip(tmp_path):
    """Verify save_pretrained/from_pretrained preserves the gate/magnitude path parameters."""
    torch.manual_seed(20)
    sae = GatedSAEModel(GatedSAEConfig(d_in=8, d_hidden=16))
    sae.save_pretrained(tmp_path)

    reloaded = GatedSAEModel.from_pretrained(tmp_path)
    assert torch.equal(reloaded.W_gate, sae.W_gate)
    assert torch.equal(reloaded.r_mag, sae.r_mag)
    assert torch.equal(reloaded.b_gate, sae.b_gate)
    assert torch.equal(reloaded.b_mag, sae.b_mag)
